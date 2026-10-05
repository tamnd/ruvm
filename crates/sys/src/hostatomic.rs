// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host atomic operations on guest memory held as `&[AtomicU8]`, what QEMU's
//! `include/qemu/atomic.h` and `atomic128.h` do on a host pointer into guest RAM.
//!
//! Every function takes the bytes of exactly one access, as a slice of 1, 2, 4, 8 or 16 bytes,
//! and does one host atomic operation of that width on them. Values are the little-endian
//! reading of the bytes, whatever the host byte order, so a caller that knows the guest byte
//! order only has to swap for big-endian accesses. A slice that is not naturally aligned in host
//! memory, or of another length, gets `None`, and the caller has to fall back to something that
//! stops the other vCPUs, as QEMU does for accesses it cannot do with one host atomic.
//!
//! Plain [`load`] and [`store`] are single-copy atomic: another thread never sees half of one.
//! Read-modify-write operations are sequentially consistent, like the `__ATOMIC_SEQ_CST`
//! builtins QEMU's atomic helpers use.
//!
//! The same bytes are also reached one byte at a time through the `AtomicU8` view, and by
//! generated code. Mixing access sizes on one location is outside the C++ and Rust memory
//! models, as it is in QEMU, but every host this runs on gives the result the guest expects:
//! an aligned access of any width is one indivisible access to its bytes.

use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64, Ordering};

/// An atomic integer that can be viewed over aligned guest bytes.
trait Cell {
    fn load(&self, o: Ordering) -> u64;
    fn store(&self, v: u64, o: Ordering);
    fn cas(&self, old: u64, new: u64) -> Result<u64, u64>;
}

macro_rules! cell {
    ($($t:ty => $u:ty),*) => {$(
        impl Cell for $t {
            fn load(&self, o: Ordering) -> u64 {
                <$u>::from_le(<$t>::load(self, o)) as u64
            }
            fn store(&self, v: u64, o: Ordering) {
                <$t>::store(self, (v as $u).to_le(), o)
            }
            fn cas(&self, old: u64, new: u64) -> Result<u64, u64> {
                let (old, new) = ((old as $u).to_le(), (new as $u).to_le());
                <$t>::compare_exchange(self, old, new, Ordering::SeqCst, Ordering::SeqCst)
                    .map(|v| <$u>::from_le(v) as u64)
                    .map_err(|v| <$u>::from_le(v) as u64)
            }
        }
    )*};
}

cell!(AtomicU8 => u8, AtomicU16 => u16, AtomicU32 => u32, AtomicU64 => u64);

/// The bytes as one `T`, if they are exactly its size and aligned for it.
fn view<T: Cell>(mem: &[AtomicU8]) -> Option<&T> {
    let p = mem.as_ptr();
    if mem.len() != size_of::<T>() || (p as usize) % align_of::<T>() != 0 {
        return None;
    }
    // SAFETY: `T` is one of the atomic integer types (the trait is private and only implemented
    // for them), which have no invalid bit patterns and the size of `T` in bytes. The slice is
    // exactly that long, so the bytes are valid for reads and writes of a `T` for the lifetime
    // of `mem`, and the address was checked to be aligned for `T`. The memory is only ever
    // reached through atomic types, so shared access through a `&T` is no more of a data race
    // than the `&[AtomicU8]` it came from.
    Some(unsafe { &*p.cast::<T>() })
}

/// Run `f` on the bytes as the atomic integer of their size.
fn with<R>(mem: &[AtomicU8], f: impl FnOnce(&dyn Cell) -> R) -> Option<R> {
    match mem.len() {
        1 => view::<AtomicU8>(mem).map(|c| f(c)),
        2 => view::<AtomicU16>(mem).map(|c| f(c)),
        4 => view::<AtomicU32>(mem).map(|c| f(c)),
        8 => view::<AtomicU64>(mem).map(|c| f(c)),
        _ => None,
    }
}

/// A single-copy atomic load of 1, 2, 4 or 8 aligned bytes, `qatomic_read`.
pub fn load(mem: &[AtomicU8]) -> Option<u64> {
    with(mem, |c| c.load(Ordering::Relaxed))
}

/// A single-copy atomic store of the low bytes of `val` to 1, 2, 4 or 8 aligned bytes,
/// `qatomic_set`. Returns false, having stored nothing, if the bytes cannot be stored at once.
pub fn store(mem: &[AtomicU8], val: u64) -> bool {
    with(mem, |c| c.store(val, Ordering::Relaxed)).is_some()
}

/// `qatomic_cmpxchg` on 1, 2, 4 or 8 aligned bytes: store `new` if they hold `old`. Returns the
/// value they held before. Only the low bytes of `old` and `new` are used.
pub fn cmpxchg(mem: &[AtomicU8], old: u64, new: u64) -> Option<u64> {
    let mask = mask(mem.len());
    with(mem, |c| match c.cas(old & mask, new & mask) {
        Ok(v) | Err(v) => v,
    })
}

/// Replace the value of 1, 2, 4 or 8 aligned bytes with `f` of it, atomically, retrying with
/// compare and swap until no other thread changed it in between. Returns the old value. This is
/// how QEMU's `atomic_template.h` does the operations a host has no single instruction for.
pub fn fetch_update(mem: &[AtomicU8], mut f: impl FnMut(u64) -> u64) -> Option<u64> {
    let mask = mask(mem.len());
    with(mem, |c| {
        let mut old = c.load(Ordering::Relaxed);
        loop {
            match c.cas(old, f(old) & mask) {
                Ok(v) => return v,
                Err(v) => old = v,
            }
        }
    })
}

fn mask(len: usize) -> u64 {
    if len >= 8 { u64::MAX } else { (1u64 << (8 * len)) - 1 }
}

/// Whether this host has a 16-byte compare and swap, QEMU's `HAVE_CMPXCHG128`.
pub fn has_cmpxchg128() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static HAS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *HAS.get_or_init(|| std::arch::is_x86_feature_detected!("cmpxchg16b"))
    }
    #[cfg(target_arch = "aarch64")]
    {
        true
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

/// `atomic16_cmpxchg` on 16 bytes aligned to 16: store `new` if they hold `old`. Returns the
/// value they held before, or `None` if the bytes are not 16 aligned ones or the host has no
/// 16-byte compare and swap.
pub fn cmpxchg128(mem: &[AtomicU8], old: u128, new: u128) -> Option<u128> {
    let p = mem.as_ptr();
    if mem.len() != 16 || (p as usize) % 16 != 0 || !has_cmpxchg128() {
        return None;
    }
    let (old, new) = (old.to_le(), new.to_le());
    Some(u128::from_le(cas16(p, old, new)))
}

/// `atomic16_read_rw`: a 16-byte single-copy atomic load, done as a compare and swap that
/// stores back what it found, so the bytes must be writable.
pub fn load128(mem: &[AtomicU8]) -> Option<u128> {
    cmpxchg128(mem, 0, 0)
}

/// Replace 16 aligned bytes with `f` of their value, atomically. Returns the old value.
pub fn fetch_update128(mem: &[AtomicU8], mut f: impl FnMut(u128) -> u128) -> Option<u128> {
    let mut old = load128(mem)?;
    loop {
        let v = cmpxchg128(mem, old, f(old))?;
        if v == old {
            return Some(v);
        }
        old = v;
    }
}

/// `lock cmpxchg16b` on the 16 aligned bytes at `p`, with the values in host order.
#[cfg(target_arch = "x86_64")]
fn cas16(p: *const AtomicU8, old: u128, new: u128) -> u128 {
    let (mut lo, mut hi) = (old as u64, (old >> 64) as u64);
    // SAFETY: the caller checked that `p` is the start of 16 bytes aligned to 16 that it holds
    // as `&[AtomicU8]`, so they are valid for an atomic read and write, and that the CPU has
    // cmpxchg16b. rbx cannot be named as an operand, so the low half of `new` is swapped into it
    // around the instruction and rbx is restored before the block ends. The instruction touches
    // only those 16 bytes, rax, rdx, rbx, rcx and the flags, all declared.
    unsafe {
        std::arch::asm!(
            "xchg {nlo}, rbx",
            "lock cmpxchg16b xmmword ptr [{p}]",
            "mov rbx, {nlo}",
            p = in(reg) p,
            nlo = inout(reg) new as u64 => _,
            in("rcx") (new >> 64) as u64,
            inout("rax") lo,
            inout("rdx") hi,
            options(nostack),
        );
    }
    lo as u128 | (hi as u128) << 64
}

/// A load-acquire, store-release exclusive pair loop on the 16 aligned bytes at `p`, with the
/// values in host order. It needs no LSE, and a failed compare stores back the value it read
/// so the exclusive monitor is released the way GCC's `__atomic_compare_exchange_16` does.
#[cfg(target_arch = "aarch64")]
fn cas16(p: *const AtomicU8, old: u128, new: u128) -> u128 {
    let (lo, hi): (u64, u64);
    // SAFETY: the caller checked that `p` is the start of 16 bytes aligned to 16 that it holds
    // as `&[AtomicU8]`, so they are valid for an atomic read and write. The loop touches only
    // those 16 bytes, the declared registers and the flags.
    unsafe {
        std::arch::asm!(
            "2:",
            "ldaxp {lo}, {hi}, [{p}]",
            "cmp {lo}, {olo}",
            "ccmp {hi}, {ohi}, #0, eq",
            "b.ne 3f",
            "stlxp {st:w}, {nlo}, {nhi}, [{p}]",
            "cbnz {st:w}, 2b",
            "b 4f",
            "3:",
            "stlxp {st:w}, {lo}, {hi}, [{p}]",
            "cbnz {st:w}, 2b",
            "4:",
            p = in(reg) p,
            olo = in(reg) old as u64,
            ohi = in(reg) (old >> 64) as u64,
            nlo = in(reg) new as u64,
            nhi = in(reg) (new >> 64) as u64,
            lo = out(reg) lo,
            hi = out(reg) hi,
            st = out(reg) _,
            options(nostack),
        );
    }
    lo as u128 | (hi as u128) << 64
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn cas16(_p: *const AtomicU8, _old: u128, _new: u128) -> u128 {
    unreachable!("has_cmpxchg128() is false on this host")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 64 bytes aligned to 64, as atomics.
    #[repr(align(64))]
    struct Buf([AtomicU8; 64]);

    fn buf() -> Buf {
        Buf(std::array::from_fn(|_| AtomicU8::new(0)))
    }

    fn bytes(b: &Buf, at: usize, n: usize) -> Vec<u8> {
        b.0[at..at + n].iter().map(|x| x.load(Ordering::Relaxed)).collect()
    }

    #[test]
    fn widths_are_little_endian() {
        let b = buf();
        assert!(store(&b.0[8..16], 0x0102_0304_0506_0708));
        assert_eq!(bytes(&b, 8, 8), [8, 7, 6, 5, 4, 3, 2, 1]);
        assert_eq!(load(&b.0[8..12]), Some(0x0506_0708));
        assert_eq!(load(&b.0[12..14]), Some(0x0304));
        assert_eq!(load(&b.0[15..16]), Some(1));
        assert!(store(&b.0[4..8], 0xdead_beef_1234_5678));
        assert_eq!(bytes(&b, 4, 4), [0x78, 0x56, 0x34, 0x12]);
    }

    #[test]
    fn misaligned_or_odd_sizes_are_refused() {
        let b = buf();
        assert_eq!(load(&b.0[1..3]), None);
        assert_eq!(load(&b.0[2..6]), None);
        assert_eq!(load(&b.0[4..12]), None);
        assert_eq!(load(&b.0[0..3]), None);
        assert!(!store(&b.0[1..5], 1));
        assert_eq!(cmpxchg(&b.0[3..5], 0, 1), None);
        assert_eq!(cmpxchg128(&b.0[8..24], 0, 1), None);
        assert_eq!(cmpxchg128(&b.0[0..8], 0, 1), None);
    }

    #[test]
    fn cmpxchg_swaps_only_on_a_match() {
        let b = buf();
        assert!(store(&b.0[0..4], 5));
        assert_eq!(cmpxchg(&b.0[0..4], 4, 9), Some(5));
        assert_eq!(load(&b.0[0..4]), Some(5));
        assert_eq!(cmpxchg(&b.0[0..4], 5, 9), Some(5));
        assert_eq!(load(&b.0[0..4]), Some(9));
        // Only the low bytes of the operands count.
        assert_eq!(cmpxchg(&b.0[0..2], 0xffff_0009, 0x1_0007), Some(9));
        assert_eq!(load(&b.0[0..4]), Some(7));
    }

    #[test]
    fn fetch_update_returns_the_old_value() {
        let b = buf();
        assert!(store(&b.0[0..1], 0xff));
        assert_eq!(fetch_update(&b.0[0..1], |v| v + 1), Some(0xff));
        assert_eq!(load(&b.0[0..1]), Some(0));
        assert_eq!(bytes(&b, 1, 1), [0]);
    }

    #[test]
    fn cmpxchg128_on_aligned_bytes() {
        if !has_cmpxchg128() {
            return;
        }
        let b = buf();
        let v = 0x0f0e_0d0c_0b0a_0908_0706_0504_0302_0100u128;
        assert_eq!(cmpxchg128(&b.0[16..32], 0, v), Some(0));
        assert_eq!(bytes(&b, 16, 16), (0..16).collect::<Vec<u8>>());
        assert_eq!(cmpxchg128(&b.0[16..32], 1, 2), Some(v));
        assert_eq!(load128(&b.0[16..32]), Some(v));
        assert_eq!(fetch_update128(&b.0[16..32], |x| x ^ !0), Some(v));
        assert_eq!(load128(&b.0[16..32]), Some(!v));
    }

    #[test]
    fn concurrent_increments_and_stores_are_not_lost() {
        // Each thread owns one byte of a shared word: it increments its byte of the word
        // with compare and swap, and stores its own byte with plain stores in between. A
        // read-modify-write done as a separate load and store would undo other threads' work.
        let b = std::sync::Arc::new(buf());
        let threads: Vec<_> = (0..4)
            .map(|t| {
                let b = b.clone();
                std::thread::spawn(move || {
                    for _ in 0..20_000 {
                        fetch_update(&b.0[0..8], |v| v.wrapping_add(1 << (16 * t)));
                        let mine = &b.0[8 + 2 * t..10 + 2 * t];
                        let v = load(mine).unwrap();
                        assert!(store(mine, v + 1));
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        let sum = load(&b.0[0..8]).unwrap();
        for t in 0..4 {
            assert_eq!((sum >> (16 * t)) & 0xffff, 20_000);
            assert_eq!(load(&b.0[8 + 2 * t..10 + 2 * t]), Some(20_000));
        }
    }
}
