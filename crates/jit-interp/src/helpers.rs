// SPDX-License-Identifier: MIT OR Apache-2.0

//! The helper registry: Rust functions the interpreter calls for `call` ops.
//!
//! Every helper is registered under the name the IR uses, with the return and argument types it
//! was declared with. A call op is checked against that declaration before the function runs.
//! Arguments arrive one 64-bit slot per value (two for an I128, low half first); I32 values are
//! zero extended. The result is returned as a `u128` whose low slots are used.
//!
//! [`HelperRegistry::new`] comes with the runtime helpers the builder itself emits:
//! `lookup_tb_ptr`, `exit_atomic` and the `atomic_*` family. The atomic helpers are not atomic:
//! the interpreter runs one thread, so a load followed by a store is enough.

use std::collections::HashMap;
use std::fmt;

use ruvm_jit_core::ir::{HelperInfo, HelperType};
use ruvm_jit_core::types::MemOpIdx;

use crate::mem::{GuestMemory, MemFault, guest_load, guest_store, plain};

/// Why control left a translation block early.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Unwind {
    /// A guest memory access failed.
    Mem(MemFault),
    /// A helper raised a guest exception with this code, as `cpu_loop_exit` would.
    Exception(u64),
    /// `exit_atomic`: the block must be run again with exclusive access.
    ExitAtomic,
}

/// What a helper can reach besides its arguments.
pub struct HelperEnv<'a> {
    /// The CPU state buffer. The `env` argument of a helper is offset 0 into it.
    pub env: &'a mut [u8],
    /// Guest memory.
    pub mem: &'a mut dyn GuestMemory,
}

impl fmt::Debug for HelperEnv<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HelperEnv").field("env_len", &self.env.len()).finish_non_exhaustive()
    }
}

/// A helper implementation.
pub type HelperFn = fn(&mut HelperEnv<'_>, &[u64]) -> Result<u128, Unwind>;

/// A registered helper.
#[derive(Clone, Debug)]
pub struct HelperEntry {
    /// The declared return type.
    pub ret: HelperType,
    /// The declared argument types.
    pub args: Vec<HelperType>,
    /// The implementation.
    pub f: HelperFn,
}

/// Helpers by name.
#[derive(Clone, Debug, Default)]
pub struct HelperRegistry {
    map: HashMap<String, HelperEntry>,
}

impl HelperRegistry {
    /// A registry holding only the built-in runtime helpers.
    pub fn new() -> HelperRegistry {
        let mut r = HelperRegistry::empty();
        r.add_builtins();
        r
    }

    /// A registry with nothing in it.
    pub fn empty() -> HelperRegistry {
        HelperRegistry { map: HashMap::new() }
    }

    /// Register `f` under `name`, replacing any earlier helper of that name.
    pub fn register(&mut self, name: &str, ret: HelperType, args: &[HelperType], f: HelperFn) {
        self.map.insert(name.to_string(), HelperEntry { ret, args: args.to_vec(), f });
    }

    /// Register `f` with the name and signature of a [`HelperInfo`].
    pub fn register_info(&mut self, info: &HelperInfo, f: HelperFn) {
        self.register(&info.name, info.ret, &info.args, f);
    }

    /// The helper registered under `name`.
    pub fn get(&self, name: &str) -> Option<&HelperEntry> {
        self.map.get(name)
    }

    /// Number of registered helpers.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// True if nothing is registered.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    fn add_builtins(&mut self) {
        use HelperType::{I32, I64, I128, Ptr, Void};
        self.register("lookup_tb_ptr", Ptr, &[Ptr], |_, _| Ok(0));
        self.register("exit_atomic", Void, &[Ptr], |_, _| Err(Unwind::ExitAtomic));

        const SUFFIXES: [(&str, u32); 9] = [
            ("b", 0),
            ("w_le", 1),
            ("w_be", 1),
            ("l_le", 2),
            ("l_be", 2),
            ("q_le", 3),
            ("q_be", 3),
            ("o_le", 4),
            ("o_be", 4),
        ];
        let rmw: [(&str, HelperFn); 17] = [
            ("fetch_add", |e, a| rmw(e, a, Rmw::Add, false)),
            ("fetch_and", |e, a| rmw(e, a, Rmw::And, false)),
            ("fetch_or", |e, a| rmw(e, a, Rmw::Or, false)),
            ("fetch_xor", |e, a| rmw(e, a, Rmw::Xor, false)),
            ("fetch_smin", |e, a| rmw(e, a, Rmw::Smin, false)),
            ("fetch_umin", |e, a| rmw(e, a, Rmw::Umin, false)),
            ("fetch_smax", |e, a| rmw(e, a, Rmw::Smax, false)),
            ("fetch_umax", |e, a| rmw(e, a, Rmw::Umax, false)),
            ("add_fetch", |e, a| rmw(e, a, Rmw::Add, true)),
            ("and_fetch", |e, a| rmw(e, a, Rmw::And, true)),
            ("or_fetch", |e, a| rmw(e, a, Rmw::Or, true)),
            ("xor_fetch", |e, a| rmw(e, a, Rmw::Xor, true)),
            ("smin_fetch", |e, a| rmw(e, a, Rmw::Smin, true)),
            ("umin_fetch", |e, a| rmw(e, a, Rmw::Umin, true)),
            ("smax_fetch", |e, a| rmw(e, a, Rmw::Smax, true)),
            ("umax_fetch", |e, a| rmw(e, a, Rmw::Umax, true)),
            ("xchg", |e, a| rmw(e, a, Rmw::Xchg, false)),
        ];
        for (suffix, size) in SUFFIXES {
            let v = match size {
                0..=2 => I32,
                3 => I64,
                _ => I128,
            };
            self.register(&format!("atomic_cmpxchg{suffix}"), v, &[Ptr, I64, v, v, I32], cmpxchg);
            for (name, f) in rmw {
                self.register(&format!("atomic_{name}{suffix}"), v, &[Ptr, I64, v, I32], f);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rmw {
    Add,
    And,
    Or,
    Xor,
    Smin,
    Umin,
    Smax,
    Umax,
    Xchg,
}

fn value(args: &[u64], i: usize, wide: bool) -> u128 {
    if wide { args[i] as u128 | (args[i + 1] as u128) << 64 } else { args[i] as u128 }
}

fn sext(v: u128, bits: u32) -> i128 {
    let sh = 128 - bits;
    ((v << sh) as i128) >> sh
}

fn rmw(e: &mut HelperEnv<'_>, args: &[u64], op: Rmw, new_value: bool) -> Result<u128, Unwind> {
    let wide = args.len() == 5;
    let addr = args[1];
    let val = value(args, 2, wide);
    let oi = plain(MemOpIdx(args[args.len() - 1] as u32));
    let bits = 8 * oi.memop().size_bytes();
    let mask = if bits == 128 { !0u128 } else { (1u128 << bits) - 1 };
    let val = val & mask;
    let old = guest_load(e.mem, addr, oi).map_err(Unwind::Mem)? & mask;
    let new = match op {
        Rmw::Add => old.wrapping_add(val),
        Rmw::And => old & val,
        Rmw::Or => old | val,
        Rmw::Xor => old ^ val,
        Rmw::Smin => {
            if sext(old, bits) <= sext(val, bits) {
                old
            } else {
                val
            }
        }
        Rmw::Umin => old.min(val),
        Rmw::Smax => {
            if sext(old, bits) >= sext(val, bits) {
                old
            } else {
                val
            }
        }
        Rmw::Umax => old.max(val),
        Rmw::Xchg => val,
    } & mask;
    guest_store(e.mem, addr, new, oi).map_err(Unwind::Mem)?;
    Ok(if new_value { new } else { old })
}

fn cmpxchg(e: &mut HelperEnv<'_>, args: &[u64]) -> Result<u128, Unwind> {
    let wide = args.len() == 7;
    let addr = args[1];
    let oi = plain(MemOpIdx(args[args.len() - 1] as u32));
    let bits = 8 * oi.memop().size_bytes();
    let mask = if bits == 128 { !0u128 } else { (1u128 << bits) - 1 };
    let (cmpv, newv) = if wide {
        (value(args, 2, true), value(args, 4, true))
    } else {
        (value(args, 2, false) & mask, value(args, 3, false) & mask)
    };
    let old = guest_load(e.mem, addr, oi).map_err(Unwind::Mem)? & mask;
    if old == cmpv {
        guest_store(e.mem, addr, newv, oi).map_err(Unwind::Mem)?;
    }
    Ok(old)
}
