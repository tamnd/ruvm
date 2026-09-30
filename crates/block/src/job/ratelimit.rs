// SPDX-License-Identifier: GPL-2.0-or-later

//! `RateLimit` from include/qemu/ratelimit.h and `ProgressMeter` from
//! include/qemu/progress_meter.h.

use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::Instant;

/// `QEMU_CLOCK_REALTIME` in nanoseconds, from an arbitrary start.
pub(crate) fn clock_ns() -> i64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_nanos() as i64
}

#[derive(Debug, Default)]
struct Slice {
    start: i64,
    end: i64,
    quota: u64,
    ns: u64,
    dispatched: u64,
}

/// `RateLimit`: a byte budget per time slice.
#[derive(Debug, Default)]
pub(crate) struct RateLimit(Mutex<Slice>);

impl RateLimit {
    /// `ratelimit_calculate_delay()`: accounts for `n` bytes and returns how many nanoseconds
    /// to wait before the next request, 0 if none.
    pub(crate) fn calculate_delay(&self, n: u64) -> i64 {
        let now = clock_ns();
        let mut l = self.0.lock().unwrap();
        if l.quota == 0 {
            // Throttling disabled.
            return 0;
        }
        if l.end < now {
            // Previous, possibly extended, time slice finished; reset the accounting.
            l.start = now;
            l.end = now + l.ns as i64;
            l.dispatched = 0;
        }
        l.dispatched += n;
        if l.dispatched < l.quota {
            // We may send further data within the current time slice, no need to delay the
            // next request.
            return 0;
        }
        // Quota exceeded. Wait based on the excess amount and then start a new slice.
        let delay_slices = l.dispatched as f64 / l.quota as f64;
        l.end = l.start + (delay_slices * l.ns as f64) as i64;
        l.end - now
    }

    /// `ratelimit_set_speed()`: `speed` bytes per second in slices of `slice_ns`; 0 turns the
    /// limit off.
    pub(crate) fn set_speed(&self, speed: u64, slice_ns: u64) {
        let mut l = self.0.lock().unwrap();
        l.ns = slice_ns;
        l.quota = if speed == 0 {
            0
        } else {
            ((speed as f64 * slice_ns as f64) / 1_000_000_000.0).max(1.0) as u64
        };
    }
}

/// `ProgressMeter`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Progress {
    pub current: u64,
    pub total: u64,
}

impl Progress {
    /// `progress_work_done()`.
    pub(crate) fn work_done(&mut self, done: u64) {
        self.current += done;
    }

    /// `progress_set_remaining()`.
    pub(crate) fn set_remaining(&mut self, remaining: u64) {
        self.total = self.current + remaining;
    }

    /// `progress_increase_remaining()`.
    pub(crate) fn increase_remaining(&mut self, delta: u64) {
        self.total += delta;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratelimit() {
        let r = RateLimit::default();
        assert_eq!(r.calculate_delay(1 << 20), 0, "no limit set");
        // 1 MiB/s in 100 ms slices is a quota of 104857 bytes.
        r.set_speed(1 << 20, 100_000_000);
        assert_eq!(r.calculate_delay(1000), 0);
        let d = r.calculate_delay(1 << 20);
        // About ten slices' worth of waiting.
        assert!(d > 900_000_000 && d <= 1_001_000_000, "{d}");
        r.set_speed(0, 100_000_000);
        assert_eq!(r.calculate_delay(1 << 30), 0);
        r.set_speed(1, 100_000_000);
        assert_eq!(r.0.lock().unwrap().quota, 1, "at least one byte per slice");
    }

    #[test]
    fn progress() {
        let mut p = Progress::default();
        p.set_remaining(100);
        p.work_done(30);
        assert_eq!(p, Progress { current: 30, total: 100 });
        p.set_remaining(10);
        assert_eq!(p.total, 40);
        p.increase_remaining(5);
        assert_eq!(p.total, 45);
    }
}
