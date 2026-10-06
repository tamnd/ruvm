// SPDX-License-Identifier: MIT OR Apache-2.0

//! The helper registry: Rust functions the interpreter calls for `call` ops.
//!
//! Every helper is registered under the name the IR uses, with the return and argument types it
//! was declared with. A call op is checked against that declaration before the function runs.
//! Arguments arrive one 64-bit slot per value (two for an I128, low half first); I32 values are
//! zero extended. The result is returned as a `u128` whose low slots are used.
//!
//! [`HelperRegistry::new`] comes with the runtime helpers the builder itself emits:
//! `lookup_tb_ptr`, `exit_atomic` and the `atomic_*` family. The atomic helpers do one host
//! atomic operation on the bytes [`GuestMemory::atomic_access`] gives them, as QEMU's do on the
//! host address of guest RAM, so they stay atomic against plain stores from other threads. A
//! memory with no host bytes to give (a test memory, say) gets a load and a store between
//! [`GuestMemory::atomic_begin`] and [`GuestMemory::atomic_end`] instead.

use std::fmt;
use std::sync::atomic::AtomicU8;

use ruvm_jit_core::hash::FastHashMap;
use ruvm_jit_core::ir::{HelperInfo, HelperType};
use ruvm_jit_core::types::MemOpIdx;
use ruvm_sys::hostatomic;

use crate::mem::{GuestMemory, MemFault, guest_load_env, guest_store_env, plain};

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

/// A helper without side effects in the host's C calling convention, which generated code may
/// call directly instead of going through the service routine: up to four argument words in,
/// one word out, unused arguments being anything. It gets neither the CPU state nor guest
/// memory, so only `TCG_CALL_NO_SE` helpers that need neither can have one. Not in QEMU, where
/// every helper is a C function to begin with.
pub type NativeHelperFn = extern "C" fn(u64, u64, u64, u64) -> u64;

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
    map: FastHashMap<String, HelperEntry>,
    native: FastHashMap<String, NativeHelperFn>,
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
        HelperRegistry { map: FastHashMap::default(), native: FastHashMap::default() }
    }

    /// Register `f` under `name`, replacing any earlier helper of that name.
    pub fn register(&mut self, name: &str, ret: HelperType, args: &[HelperType], f: HelperFn) {
        self.native.remove(name);
        self.map.insert(name.to_string(), HelperEntry { ret, args: args.to_vec(), f });
    }

    /// Give the helper already registered under `name` a [`NativeHelperFn`] doing the same
    /// work, for backends that call helpers directly. Ignored when nothing is registered under
    /// `name`, or when it is not `I64`, `I32` or `Void` valued with at most four `I64` or `I32`
    /// arguments; registering the helper again drops it.
    pub fn register_native(&mut self, name: &str, f: NativeHelperFn) {
        use HelperType::{I32, I64, Void};
        let Some(e) = self.map.get(name) else { return };
        let word = |t: &HelperType| matches!(t, I32 | I64);
        if (word(&e.ret) || e.ret == Void) && e.args.len() <= 4 && e.args.iter().all(word) {
            self.native.insert(name.to_string(), f);
        }
    }

    /// The [`NativeHelperFn`] of the helper registered under `name`, if it has one.
    pub fn native(&self, name: &str) -> Option<NativeHelperFn> {
        self.native.get(name).copied()
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
        self.register("lookup_tb_ptr_ic", Ptr, &[Ptr, I64], |_, _| Ok(0));
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

/// The new value of a read-modify-write `op` on `old` with operand `val`, both `bits` wide.
fn apply(op: Rmw, old: u128, val: u128, bits: u32) -> u128 {
    match op {
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
    }
}

/// The operands of an atomic helper: address, memop, width in bits and the mask of that many
/// bits.
fn operands(args: &[u64]) -> (u64, MemOpIdx, u32, u128) {
    let addr = args[1];
    let oi = plain(MemOpIdx(args[args.len() - 1] as u32));
    let bits = 8 * oi.memop().size_bytes();
    let mask = if bits == 128 { !0u128 } else { (1u128 << bits) - 1 };
    (addr, oi, bits, mask)
}

/// Convert between a value and how its `bits` are laid out in memory, little endian first.
fn swap(v: u128, bits: u32, bswap: bool) -> u128 {
    if bswap { v.swap_bytes() >> (128 - bits) } else { v }
}

/// Replace the value at `bytes` with `f` of it with host atomics. Returns the old value, or
/// `None` when the host cannot do it in one atomic operation.
fn host_update(
    bytes: &[AtomicU8],
    bits: u32,
    bswap: bool,
    mut f: impl FnMut(u128) -> u128,
) -> Option<u128> {
    let old = if bits == 128 {
        hostatomic::fetch_update128(bytes, |m| swap(f(swap(m, bits, bswap)), bits, bswap))?
    } else {
        let g = |m: u64| swap(f(swap(m as u128, bits, bswap)), bits, bswap) as u64;
        u128::from(hostatomic::fetch_update(bytes, g)?)
    };
    Some(swap(old, bits, bswap))
}

/// `cmpxchg` with host atomics, on the bytes of the access. Returns the old value, or `None`
/// when the host cannot do it in one atomic operation.
fn host_cmpxchg(
    bytes: &[AtomicU8],
    bits: u32,
    bswap: bool,
    cmpv: u128,
    newv: u128,
) -> Option<u128> {
    let (c, n) = (swap(cmpv, bits, bswap), swap(newv, bits, bswap));
    let old = if bits == 128 {
        hostatomic::cmpxchg128(bytes, c, n)?
    } else {
        u128::from(hostatomic::cmpxchg(bytes, c as u64, n as u64)?)
    };
    Some(swap(old, bits, bswap))
}

/// `atomic_<op>` helpers: a read-modify-write done as one host atomic operation on guest
/// memory when the memory can give its bytes, the way `atomic_template.h` works on the host
/// address `atomic_mmu_lookup()` returns.
fn rmw(e: &mut HelperEnv<'_>, args: &[u64], op: Rmw, new_value: bool) -> Result<u128, Unwind> {
    let wide = args.len() == 5;
    let (addr, oi, bits, mask) = operands(args);
    let val = value(args, 2, wide) & mask;
    let bswap = oi.memop().is_bswap();
    let mut old = None;
    let mut run = |bytes: &[AtomicU8]| {
        old = host_update(bytes, bits, bswap, |v| apply(op, v, val, bits) & mask);
    };
    if e.mem.atomic_access(e.env, addr, oi, &mut run)? {
        let old = old.ok_or(Unwind::ExitAtomic)?;
        return Ok(if new_value { apply(op, old, val, bits) & mask } else { old });
    }
    e.mem.atomic_begin();
    let r = rmw_locked(e, args, op, new_value);
    e.mem.atomic_end();
    r
}

/// [`rmw`] for a memory without host bytes: a load and a store under the memory's lock.
fn rmw_locked(
    e: &mut HelperEnv<'_>,
    args: &[u64],
    op: Rmw,
    new_value: bool,
) -> Result<u128, Unwind> {
    let wide = args.len() == 5;
    let (addr, oi, bits, mask) = operands(args);
    let val = value(args, 2, wide) & mask;
    let old = guest_load_env(e.mem, e.env, addr, oi).map_err(Unwind::Mem)? & mask;
    let new = apply(op, old, val, bits) & mask;
    guest_store_env(e.mem, e.env, addr, new, oi).map_err(Unwind::Mem)?;
    Ok(if new_value { new } else { old })
}

/// `atomic_cmpxchg` helpers, with host atomics as [`rmw`] does.
fn cmpxchg(e: &mut HelperEnv<'_>, args: &[u64]) -> Result<u128, Unwind> {
    let wide = args.len() == 7;
    let (addr, oi, bits, mask) = operands(args);
    let (cmpv, newv) = if wide {
        (value(args, 2, true), value(args, 4, true))
    } else {
        (value(args, 2, false) & mask, value(args, 3, false) & mask)
    };
    let bswap = oi.memop().is_bswap();
    let mut old = None;
    let mut run = |bytes: &[AtomicU8]| old = host_cmpxchg(bytes, bits, bswap, cmpv, newv);
    if e.mem.atomic_access(e.env, addr, oi, &mut run)? {
        return old.ok_or(Unwind::ExitAtomic);
    }
    e.mem.atomic_begin();
    let r = cmpxchg_locked(e, args);
    e.mem.atomic_end();
    r
}

/// [`cmpxchg`] for a memory without host bytes: a load and a store under the memory's lock.
fn cmpxchg_locked(e: &mut HelperEnv<'_>, args: &[u64]) -> Result<u128, Unwind> {
    let wide = args.len() == 7;
    let (addr, oi, _, mask) = operands(args);
    let (cmpv, newv) = if wide {
        (value(args, 2, true), value(args, 4, true))
    } else {
        (value(args, 2, false) & mask, value(args, 3, false) & mask)
    };
    let old = guest_load_env(e.mem, e.env, addr, oi).map_err(Unwind::Mem)? & mask;
    if old == cmpv {
        guest_store_env(e.mem, e.env, addr, newv, oi).map_err(Unwind::Mem)?;
    }
    Ok(old)
}

#[cfg(test)]
mod tests {
    use super::*;

    extern "C" fn sum(a: u64, b: u64, c: u64, d: u64) -> u64 {
        a + b + c + d
    }

    fn zero(_: &mut HelperEnv<'_>, _: &[u64]) -> Result<u128, Unwind> {
        Ok(0)
    }

    #[test]
    fn native_entry_points_follow_their_helper() {
        use HelperType::{I32, I64, I128, Ptr, Void};
        let mut r = HelperRegistry::empty();
        // Nothing registered under the name.
        r.register_native("a", sum);
        assert!(r.native("a").is_none());
        // Word valued helpers with up to four word arguments get one.
        r.register("a", I64, &[I64, I32, I64, I32], zero);
        r.register_native("a", sum);
        assert_eq!(r.native("a").map(|f| f(1, 2, 3, 4)), Some(10));
        r.register("v", Void, &[], zero);
        r.register_native("v", sum);
        assert!(r.native("v").is_some());
        // Registering the helper again drops it.
        r.register("a", I64, &[I64], zero);
        assert!(r.native("a").is_none());
        // Other signatures do not.
        for (name, ret, args) in [
            ("p", I64, &[Ptr][..]),
            ("w", I128, &[I64][..]),
            ("five", I32, &[I32, I32, I32, I32, I32][..]),
        ] {
            r.register(name, ret, args, zero);
            r.register_native(name, sum);
            assert!(r.native(name).is_none(), "{name}");
        }
    }
}
