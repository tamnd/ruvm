// SPDX-License-Identifier: MIT OR Apache-2.0

//! The host's cycle counter, what QEMU's `cpu_get_host_ticks()` in `include/qemu/timer.h` reads.
//!
//! QEMU's TCG gives an x86 guest a TSC that is `cpu_get_ticks()`, and that counts host ticks: the
//! host's own TSC on an x86 host, so the guest TSC runs at the host TSC rate. A guest kernel
//! calibrates its clock against that rate, and a migrated guest keeps the rate it calibrated
//! with, so ruvm has to count at the same rate as QEMU on the same host for a guest to keep time
//! when it moves between them.
//!
//! On a host other than x86_64 this counts nanoseconds of the monotonic clock since the first
//! call, what QEMU's generic `cpu_get_host_ticks()` returns through `get_clock()`.
//!
//! [`Ticks`] is `cpu_get_ticks()` itself: host ticks that only count while the guest runs, so
//! that a stopped or migrating guest's TSC stops with its virtual clock.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering, fence};
use std::sync::{Mutex, PoisonError};

/// The host's tick counter: `rdtsc` on x86_64, monotonic nanoseconds elsewhere. Only the
/// difference between two readings means anything.
#[cfg(target_arch = "x86_64")]
pub fn host_ticks() -> u64 {
    let (lo, hi): (u32, u32);
    // SAFETY: rdtsc reads the time stamp counter into edx:eax and touches nothing else; it does
    // not fault in user mode on the hosts ruvm runs on, where CR4.TSD is clear.
    unsafe {
        std::arch::asm!("rdtsc", out("eax") lo, out("edx") hi, options(nomem, nostack, preserves_flags));
    }
    (u64::from(hi) << 32) | u64::from(lo)
}

/// The host's tick counter: `rdtsc` on x86_64, monotonic nanoseconds elsewhere. Only the
/// difference between two readings means anything.
#[cfg(not(target_arch = "x86_64"))]
pub fn host_ticks() -> u64 {
    use std::sync::LazyLock;
    use std::time::Instant;

    static BASE: LazyLock<Instant> = LazyLock::new(Instant::now);
    BASE.elapsed().as_nanos() as u64
}

/// `cpu_get_ticks()` over `timers_state` in QEMU's `system/cpu-timers.c`: host ticks that
/// count only while enabled, from `cpu_enable_ticks()` to `cpu_disable_ticks()`. Reading never
/// blocks; the rare changes go through a sequence count, as `vm_clock_seqlock` does in QEMU.
#[derive(Debug, Default)]
pub struct Ticks {
    /// Odd while a change is being made to the two fields below.
    seq: AtomicU64,
    /// `cpu_ticks_offset`: the count itself while disabled, the count minus the host ticks
    /// while enabled.
    offset: AtomicU64,
    /// `cpu_ticks_enabled`.
    enabled: AtomicBool,
    /// Taken by whoever changes the fields, so that only one does at a time.
    write: Mutex<()>,
}

impl Ticks {
    /// A count at 0 that is not counting yet, as before `vm_start()`.
    pub fn new() -> Ticks {
        Ticks::default()
    }

    /// A count that starts at 0 now and is counting.
    pub fn running() -> Ticks {
        let t = Ticks::new();
        t.enable();
        t
    }

    /// `cpu_get_ticks()`.
    pub fn get(&self) -> u64 {
        loop {
            let seq = self.seq.load(Ordering::Acquire);
            let offset = self.offset.load(Ordering::Relaxed);
            let enabled = self.enabled.load(Ordering::Relaxed);
            fence(Ordering::Acquire);
            if seq & 1 == 0 && self.seq.load(Ordering::Relaxed) == seq {
                return if enabled { offset.wrapping_add(host_ticks()) } else { offset };
            }
            std::hint::spin_loop();
        }
    }

    /// Whether the count is going, `cpu_ticks_enabled`.
    pub fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// `cpu_enable_ticks()`: the count goes on from where it stopped. Does nothing if it is
    /// counting already.
    pub fn enable(&self) {
        self.update(|offset, enabled| {
            if enabled { (offset, true) } else { (offset.wrapping_sub(host_ticks()), true) }
        });
    }

    /// `cpu_disable_ticks()`: the count stops where it is. Does nothing if it is stopped
    /// already.
    pub fn disable(&self) {
        self.update(|offset, enabled| {
            if enabled { (offset.wrapping_add(host_ticks()), false) } else { (offset, false) }
        });
    }

    /// Sets the count to `ticks`, as loading the `timer` section sets `cpu_ticks_offset`. A
    /// count that is going goes on from `ticks`.
    pub fn set(&self, ticks: u64) {
        self.update(|_, enabled| {
            (if enabled { ticks.wrapping_sub(host_ticks()) } else { ticks }, enabled)
        });
    }

    fn update(&self, f: impl FnOnce(u64, bool) -> (u64, bool)) {
        let _write = self.write.lock().unwrap_or_else(PoisonError::into_inner);
        let (offset, enabled) =
            f(self.offset.load(Ordering::Relaxed), self.enabled.load(Ordering::Relaxed));
        self.seq.fetch_add(1, Ordering::Relaxed);
        fence(Ordering::Release);
        self.offset.store(offset, Ordering::Relaxed);
        self.enabled.store(enabled, Ordering::Relaxed);
        self.seq.fetch_add(1, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn ticks_go_forward() {
        let a = host_ticks();
        std::thread::sleep(Duration::from_millis(2));
        let b = host_ticks();
        // At least 2 ms worth at any rate above 1 MHz.
        assert!(b.wrapping_sub(a) >= 2000, "{a} then {b}");
        assert!(b.wrapping_sub(a) < 1 << 62);
    }

    #[test]
    fn ticks_count_only_while_enabled() {
        let t = Ticks::new();
        assert!(!t.enabled());
        std::thread::sleep(Duration::from_millis(2));
        assert_eq!(t.get(), 0);
        t.enable();
        assert!(t.enabled());
        std::thread::sleep(Duration::from_millis(2));
        let a = t.get();
        assert!(a >= 2000, "{a}");
        t.disable();
        let b = t.get();
        assert!(b >= a);
        std::thread::sleep(Duration::from_millis(2));
        assert_eq!(t.get(), b);
        // Twice changes nothing.
        t.disable();
        assert_eq!(t.get(), b);
        t.enable();
        t.enable();
        let c = t.get();
        assert!(c >= b && c - b < 1 << 32, "{b} then {c}");
    }

    #[test]
    fn set_moves_the_count() {
        let t = Ticks::new();
        t.set(1 << 40);
        assert_eq!(t.get(), 1 << 40);
        t.enable();
        let a = t.get();
        assert!(a >= 1 << 40 && a - (1 << 40) < 1 << 32, "{a}");
        t.set(5);
        assert!(t.get() >= 5 && t.get() < 1 << 32);
        let r = Ticks::running();
        assert!(r.enabled());
        assert!(r.get() < 1 << 40);
    }
}
