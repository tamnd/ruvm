// SPDX-License-Identifier: GPL-2.0-or-later

//! I/O throttling: the leaky bucket algorithm from util/throttle.c and throttle groups from
//! block/throttle-groups.c (in [`groups`]).
//!
//! Differences from QEMU:
//!
//! - QEMU reads a `QEMUClockType` for the current time. Here a [`ThrottleTimers`] and a group
//!   carry a [`Clock`], so tests can use a [`VirtualClock`] they move by hand. The real clock
//!   is monotonic and starts well above zero, as `QEMU_CLOCK_REALTIME` does.
//! - A timer is only a deadline ([`Timer`]). There is no event loop to fire it: the threads
//!   that wait in a throttle group fire the timers of their group when they expire (see
//!   [`groups`]). `throttle_timers_attach_aio_context()` and its detach counterpart create and
//!   drop the timers, as they do in QEMU, without an `AioContext`.

pub mod groups;

#[cfg(test)]
mod tests;

use std::fmt;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use ruvm_base::{Error, Result};
use ruvm_qapi::types::ThrottleLimits;

/// `THROTTLE_VALUE_MAX`.
pub const THROTTLE_VALUE_MAX: u64 = 1_000_000_000_000_000;

const NANOSECONDS_PER_SECOND: i64 = 1_000_000_000;

/// `BucketType`: the index of a bucket in [`ThrottleConfig::buckets`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BucketType {
    BpsTotal = 0,
    BpsRead,
    BpsWrite,
    OpsTotal,
    OpsRead,
    OpsWrite,
}

/// `BUCKETS_COUNT`.
pub const BUCKETS_COUNT: usize = 6;

impl BucketType {
    /// All the buckets, in index order.
    pub const ALL: [BucketType; BUCKETS_COUNT] = [
        BucketType::BpsTotal,
        BucketType::BpsRead,
        BucketType::BpsWrite,
        BucketType::OpsTotal,
        BucketType::OpsRead,
        BucketType::OpsWrite,
    ];
}

/// `ThrottleDirection`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThrottleDirection {
    Read = 0,
    Write = 1,
}

/// `THROTTLE_MAX`.
pub const THROTTLE_MAX: usize = 2;

impl ThrottleDirection {
    /// Both directions, in index order.
    pub const ALL: [ThrottleDirection; THROTTLE_MAX] =
        [ThrottleDirection::Read, ThrottleDirection::Write];
}

/// `LeakyBucket`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LeakyBucket {
    /// The average goal in units per second.
    pub avg: u64,
    /// The burst limit in units per second.
    pub max: u64,
    /// The bucket level in units.
    pub level: f64,
    /// The level of the bucket that enforces `max` during bursts.
    pub burst_level: f64,
    /// The longest burst, in seconds.
    pub burst_length: u64,
}

impl LeakyBucket {
    /// `throttle_leak_bucket()`: leaks what `delta_ns` of time lets through.
    pub fn leak(&mut self, delta_ns: i64) {
        let leak = (self.avg as f64 * delta_ns as f64) / NANOSECONDS_PER_SECOND as f64;
        self.level = (self.level - leak).max(0.0);

        // Bursts longer than a second need burst_level to hold the rate to max.
        if self.burst_length > 1 {
            let leak = (self.max as f64 * delta_ns as f64) / NANOSECONDS_PER_SECOND as f64;
            self.burst_level = (self.burst_level - leak).max(0.0);
        }
    }

    /// `throttle_compute_wait()`: how long in ns before the bucket lets I/O through again,
    /// 0 if it does now.
    pub fn compute_wait(&self) -> i64 {
        if self.avg == 0 {
            return 0;
        }

        let (bucket_size, burst_bucket_size) = if self.max == 0 {
            // Without a burst limit short bursts still go through, otherwise every other
            // request would be throttled.
            (self.avg as f64 / 10.0, 0.0)
        } else {
            // All I/O at the burst rate must be done before the average rate applies.
            (self.max.wrapping_mul(self.burst_length) as f64, self.max as f64 / 10.0)
        };

        // A full bucket means waiting.
        let extra = self.level - bucket_size;
        if extra > 0.0 {
            return do_compute_wait(self.avg as f64, extra);
        }

        // Otherwise the burst bucket may still hold the rate to max.
        if self.burst_length > 1 {
            debug_assert!(self.max > 0, "see throttle_is_valid()");
            let extra = self.burst_level - burst_bucket_size;
            if extra > 0.0 {
                return do_compute_wait(self.max as f64, extra);
            }
        }

        0
    }
}

/// `throttle_do_compute_wait()`.
fn do_compute_wait(limit: f64, extra: f64) -> i64 {
    let wait = extra * NANOSECONDS_PER_SECOND as f64;
    (wait / limit) as i64
}

/// `ThrottleConfig`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ThrottleConfig {
    pub buckets: [LeakyBucket; BUCKETS_COUNT],
    /// The size of an operation in bytes, 0 for any size.
    pub op_size: u64,
}

impl ThrottleConfig {
    /// `throttle_config_init()`: no limits, burst lengths of one second.
    pub fn new() -> Self {
        let mut cfg = ThrottleConfig::default();
        for b in &mut cfg.buckets {
            b.burst_length = 1;
        }
        cfg
    }

    /// The bucket `t`.
    pub fn bucket(&self, t: BucketType) -> &LeakyBucket {
        &self.buckets[t as usize]
    }

    /// The bucket `t`, to change.
    pub fn bucket_mut(&mut self, t: BucketType) -> &mut LeakyBucket {
        &mut self.buckets[t as usize]
    }

    /// `throttle_enabled()`: whether any limit is set.
    pub fn enabled(&self) -> bool {
        self.buckets.iter().any(|b| b.avg > 0)
    }

    /// `throttle_is_valid()`.
    pub fn is_valid(&self) -> Result<()> {
        use BucketType::*;
        let b = |t: BucketType| self.bucket(t);
        let bps_flag = b(BpsTotal).avg != 0 && (b(BpsRead).avg != 0 || b(BpsWrite).avg != 0);
        let ops_flag = b(OpsTotal).avg != 0 && (b(OpsRead).avg != 0 || b(OpsWrite).avg != 0);
        let bps_max_flag = b(BpsTotal).max != 0 && (b(BpsRead).max != 0 || b(BpsWrite).max != 0);
        let ops_max_flag = b(OpsTotal).max != 0 && (b(OpsRead).max != 0 || b(OpsWrite).max != 0);

        if bps_flag || ops_flag || bps_max_flag || ops_max_flag {
            return Err(Error::generic(
                "bps/iops/max total values and read/write values cannot be used at the same time",
            ));
        }

        if self.op_size != 0 && b(OpsTotal).avg == 0 && b(OpsRead).avg == 0 && b(OpsWrite).avg == 0
        {
            return Err(Error::generic("iops size requires an iops value to be set"));
        }

        for bkt in &self.buckets {
            if bkt.avg > THROTTLE_VALUE_MAX || bkt.max > THROTTLE_VALUE_MAX {
                return Err(Error::generic(format!(
                    "bps/iops/max values must be within [0, {THROTTLE_VALUE_MAX}]"
                )));
            }
            if bkt.burst_length == 0 {
                return Err(Error::generic("the burst length cannot be 0"));
            }
            if bkt.burst_length > 1 && bkt.max == 0 {
                return Err(Error::generic("burst length set without burst rate"));
            }
            if bkt.max != 0 && bkt.burst_length > THROTTLE_VALUE_MAX / bkt.max {
                return Err(Error::generic("burst length too high for this burst rate"));
            }
            if bkt.max != 0 && bkt.avg == 0 {
                return Err(Error::generic(
                    "bps_max/iops_max require corresponding bps/iops values",
                ));
            }
            if bkt.max != 0 && bkt.max < bkt.avg {
                return Err(Error::generic("bps_max/iops_max cannot be lower than bps/iops"));
            }
        }

        Ok(())
    }

    /// `throttle_limits_to_config()`: sets what `arg` has, then checks the result with
    /// [`ThrottleConfig::is_valid`]. On a range error nothing after the offending field is set.
    pub fn apply_limits(&mut self, arg: &ThrottleLimits) -> Result<()> {
        use BucketType::*;
        // The QAPI ints go into unsigned fields as C converts them.
        let set = |v: Option<i64>, f: &mut u64| {
            if let Some(v) = v {
                *f = v as u64;
            }
        };
        set(arg.bps_total, &mut self.bucket_mut(BpsTotal).avg);
        set(arg.bps_read, &mut self.bucket_mut(BpsRead).avg);
        set(arg.bps_write, &mut self.bucket_mut(BpsWrite).avg);
        set(arg.iops_total, &mut self.bucket_mut(OpsTotal).avg);
        set(arg.iops_read, &mut self.bucket_mut(OpsRead).avg);
        set(arg.iops_write, &mut self.bucket_mut(OpsWrite).avg);
        set(arg.bps_total_max, &mut self.bucket_mut(BpsTotal).max);
        set(arg.bps_read_max, &mut self.bucket_mut(BpsRead).max);
        set(arg.bps_write_max, &mut self.bucket_mut(BpsWrite).max);
        set(arg.iops_total_max, &mut self.bucket_mut(OpsTotal).max);
        set(arg.iops_read_max, &mut self.bucket_mut(OpsRead).max);
        set(arg.iops_write_max, &mut self.bucket_mut(OpsWrite).max);

        let lengths = [
            ("bps-total-max-length", arg.bps_total_max_length, BpsTotal),
            ("bps-read-max-length", arg.bps_read_max_length, BpsRead),
            ("bps-write-max-length", arg.bps_write_max_length, BpsWrite),
            ("iops-total-max-length", arg.iops_total_max_length, OpsTotal),
            ("iops-read-max-length", arg.iops_read_max_length, OpsRead),
            ("iops-write-max-length", arg.iops_write_max_length, OpsWrite),
        ];
        for (name, v, t) in lengths {
            if let Some(v) = v {
                // A signed comparison with UINT_MAX, as in C: negative values pass here.
                if v > i64::from(u32::MAX) {
                    return Err(Error::generic(format!(
                        "{name} value must be in the range [0, {}]",
                        u32::MAX
                    )));
                }
                self.bucket_mut(t).burst_length = v as u64;
            }
        }

        if let Some(v) = arg.iops_size {
            self.op_size = v as u64;
        }

        self.is_valid()
    }

    /// `throttle_config_to_limits()`: every field set.
    pub fn to_limits(&self) -> ThrottleLimits {
        use BucketType::*;
        let avg = |t| Some(self.bucket(t).avg as i64);
        let max = |t| Some(self.bucket(t).max as i64);
        let len = |t| Some(self.bucket(t).burst_length as i64);
        ThrottleLimits {
            iops_total: avg(OpsTotal),
            iops_total_max: max(OpsTotal),
            iops_total_max_length: len(OpsTotal),
            iops_read: avg(OpsRead),
            iops_read_max: max(OpsRead),
            iops_read_max_length: len(OpsRead),
            iops_write: avg(OpsWrite),
            iops_write_max: max(OpsWrite),
            iops_write_max_length: len(OpsWrite),
            bps_total: avg(BpsTotal),
            bps_total_max: max(BpsTotal),
            bps_total_max_length: len(BpsTotal),
            bps_read: avg(BpsRead),
            bps_read_max: max(BpsRead),
            bps_read_max_length: len(BpsRead),
            bps_write: avg(BpsWrite),
            bps_write_max: max(BpsWrite),
            bps_write_max_length: len(BpsWrite),
            iops_size: Some(self.op_size as i64),
        }
    }
}

/// A source of time in nanoseconds, what a `QEMUClockType` stands for.
pub trait Clock: Send + Sync + fmt::Debug {
    /// `qemu_clock_get_ns()`.
    fn now_ns(&self) -> i64;
}

/// `QEMU_CLOCK_REALTIME`: monotonic time that does not start at zero.
#[derive(Debug)]
pub struct RealtimeClock {
    start: Instant,
}

/// Where the real clock starts, so that a timestamp is never 0.
const REALTIME_BASE_NS: i64 = NANOSECONDS_PER_SECOND;

impl Clock for RealtimeClock {
    fn now_ns(&self) -> i64 {
        let d = self.start.elapsed().as_nanos();
        REALTIME_BASE_NS + i64::try_from(d).unwrap_or(i64::MAX - REALTIME_BASE_NS)
    }
}

/// The shared real clock.
pub fn realtime_clock() -> Arc<dyn Clock> {
    static CLOCK: OnceLock<Arc<RealtimeClock>> = OnceLock::new();
    CLOCK.get_or_init(|| Arc::new(RealtimeClock { start: Instant::now() })).clone()
}

/// `QEMU_CLOCK_VIRTUAL` under qtest: time only moves when told to.
#[derive(Debug, Default)]
pub struct VirtualClock {
    now: AtomicI64,
}

impl VirtualClock {
    /// A clock that reads `start`.
    pub fn new(start: i64) -> Self {
        VirtualClock { now: AtomicI64::new(start) }
    }

    /// Moves the clock forward by `ns`.
    pub fn advance(&self, ns: i64) {
        self.now.fetch_add(ns, Ordering::SeqCst);
    }

    /// Sets the clock to `ns`.
    pub fn set(&self, ns: i64) {
        self.now.store(ns, Ordering::SeqCst);
    }
}

impl Clock for VirtualClock {
    fn now_ns(&self) -> i64 {
        self.now.load(Ordering::SeqCst)
    }
}

/// A `QEMUTimer`, reduced to its deadline.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Timer {
    expire: Option<i64>,
}

impl Timer {
    /// `timer_pending()`.
    pub fn pending(&self) -> bool {
        self.expire.is_some()
    }

    /// When the timer fires, if it is armed.
    pub fn expire_time(&self) -> Option<i64> {
        self.expire
    }

    /// `timer_mod()`.
    pub fn modify(&mut self, expire: i64) {
        self.expire = Some(expire);
    }

    /// `timer_del()`.
    pub fn del(&mut self) {
        self.expire = None;
    }
}

/// `ThrottleTimers`.
#[derive(Clone, Debug)]
pub struct ThrottleTimers {
    /// `timers[direction]`, present while attached and when that direction has a callback.
    pub timers: [Option<Timer>; THROTTLE_MAX],
    /// Which directions have a callback (`timer_cb[direction]` in QEMU).
    has_cb: [bool; THROTTLE_MAX],
    /// `clock_type`.
    pub clock: Arc<dyn Clock>,
}

impl ThrottleTimers {
    /// `throttle_timers_init()`: timers for the directions that have a callback.
    pub fn new(clock: Arc<dyn Clock>, read_cb: bool, write_cb: bool) -> Self {
        let mut tt =
            ThrottleTimers { timers: [None; THROTTLE_MAX], has_cb: [read_cb, write_cb], clock };
        tt.attach();
        tt
    }

    /// `throttle_timers_attach_aio_context()`.
    pub fn attach(&mut self) {
        for d in 0..THROTTLE_MAX {
            if self.has_cb[d] {
                self.timers[d] = Some(Timer::default());
            }
        }
    }

    /// `throttle_timers_detach_aio_context()`: the timers go, armed or not.
    pub fn detach(&mut self) {
        self.timers = [None; THROTTLE_MAX];
    }

    /// `throttle_timers_destroy()`.
    pub fn destroy(&mut self) {
        self.detach();
    }

    /// `throttle_timers_are_initialized()`.
    pub fn are_initialized(&self) -> bool {
        self.timers.iter().any(Option::is_some)
    }

    /// Whether the timer of `dir` is armed.
    pub fn pending(&self, dir: ThrottleDirection) -> bool {
        self.timers[dir as usize].is_some_and(|t| t.pending())
    }
}

/// `ThrottleState`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ThrottleState {
    pub cfg: ThrottleConfig,
    /// The time of the last leak.
    pub previous_leak: i64,
}

impl ThrottleState {
    /// `throttle_init()`: no limits and no previous leak.
    pub fn new() -> Self {
        ThrottleState { cfg: ThrottleConfig::new(), previous_leak: 0 }
    }

    /// `throttle_config()`: takes `cfg` with empty buckets, from now on.
    pub fn config(&mut self, clock: &dyn Clock, cfg: &ThrottleConfig) {
        self.cfg = *cfg;
        for b in &mut self.cfg.buckets {
            b.level = 0.0;
            b.burst_level = 0.0;
        }
        self.previous_leak = clock.now_ns();
    }

    /// `throttle_get_config()`.
    pub fn get_config(&self) -> ThrottleConfig {
        self.cfg
    }

    /// `throttle_do_leak()`.
    fn do_leak(&mut self, now: i64) {
        let delta_ns = now - self.previous_leak;
        self.previous_leak = now;
        if delta_ns <= 0 {
            return;
        }
        for b in &mut self.cfg.buckets {
            b.leak(delta_ns);
        }
    }

    /// `throttle_compute_wait_for()`.
    fn compute_wait_for(&self, dir: ThrottleDirection) -> i64 {
        use BucketType::*;
        let to_check = match dir {
            ThrottleDirection::Read => [BpsTotal, OpsTotal, BpsRead, OpsRead],
            ThrottleDirection::Write => [BpsTotal, OpsTotal, BpsWrite, OpsWrite],
        };
        to_check.iter().map(|&t| self.cfg.bucket(t).compute_wait()).fold(0, i64::max)
    }

    /// `throttle_compute_timer()`: after leaking up to `now`, when the next request of
    /// `dir` may go, or `None` if it may go now.
    pub fn compute_timer(&mut self, dir: ThrottleDirection, now: i64) -> Option<i64> {
        self.do_leak(now);
        match self.compute_wait_for(dir) {
            0 => None,
            wait => Some(now + wait),
        }
    }

    /// `throttle_schedule_timer()`: whether a request of `dir` must wait. If it must and the
    /// timer of `dir` is not armed yet, it is armed for when the request may go.
    pub fn schedule_timer(&mut self, tt: &mut ThrottleTimers, dir: ThrottleDirection) -> bool {
        let now = tt.clock.now_ns();
        let timer = tt.timers[dir as usize].as_mut().expect("the throttle timer exists");
        let Some(next) = self.compute_timer(dir, now) else {
            return false;
        };
        if !timer.pending() {
            timer.modify(next);
        }
        true
    }

    /// `throttle_account()`: puts a request of `size` bytes into the buckets of `dir`.
    pub fn account(&mut self, dir: ThrottleDirection, size: u64) {
        use BucketType::*;
        let op_size = self.cfg.op_size;
        // With op_size set, a larger request counts as several operations.
        let units = if op_size != 0 && size > op_size { size as f64 / op_size as f64 } else { 1.0 };
        let (size_buckets, unit_buckets) = match dir {
            ThrottleDirection::Read => ([BpsTotal, BpsRead], [OpsTotal, OpsRead]),
            ThrottleDirection::Write => ([BpsTotal, BpsWrite], [OpsTotal, OpsWrite]),
        };
        for i in 0..2 {
            let b = self.cfg.bucket_mut(size_buckets[i]);
            b.level += size as f64;
            if b.burst_length > 1 {
                b.burst_level += size as f64;
            }
            let b = self.cfg.bucket_mut(unit_buckets[i]);
            b.level += units;
            if b.burst_length > 1 {
                b.burst_level += units;
            }
        }
    }
}
