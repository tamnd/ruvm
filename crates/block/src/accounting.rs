// SPDX-License-Identifier: GPL-2.0-or-later

//! I/O accounting for `query-blockstats`, from block/accounting.c, with the timed averages of
//! util/timed-average.c behind `stats-intervals`.
//!
//! Differences from QEMU:
//!
//! - In QEMU the devices account their requests (`block_acct_start()` in virtio-blk, IDE,
//!   SCSI). Here [`BlockBackend`](crate::BlockBackend) accounts every request made through it:
//!   reads, writes (write zeroes count as writes of their length), discards as unmap and
//!   flushes. A request that fails `blk_check_byte_request()` counts as invalid.
//! - The clock is a [`Clock`] the stats are made with, not `QEMU_CLOCK_REALTIME` or, under
//!   qtest, `QEMU_CLOCK_VIRTUAL`; the fixed qtest latency of 1 ms is not emulated.
//! - [`BlockAcctStats::query`] leaves `wr_highest_offset` at 0: it is a property of the root
//!   node, which [`BlockBackend::stats`](crate::BlockBackend::stats) fills in.

use std::io;
use std::sync::{Arc, Mutex};

use ruvm_base::{Error, Result};
use ruvm_qapi::types::{BlockDeviceStats, BlockDeviceTimedStats, BlockLatencyHistogramInfo};

use crate::throttle::{Clock, realtime_clock};

const NANOSECONDS_PER_SECOND: u64 = 1_000_000_000;

/// `enum BlockAcctType`, without `BLOCK_ACCT_NONE`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockAcctType {
    Read = 0,
    Write,
    Flush,
    ZoneAppend,
    Unmap,
}

/// `BLOCK_MAX_IOTYPE`, less the unused `BLOCK_ACCT_NONE`.
const MAX_IOTYPE: usize = 5;

/// `BlockAcctCookie`: a request being accounted, from [`BlockAcctStats::start`] to
/// [`BlockAcctStats::done`] or [`BlockAcctStats::failed`].
#[derive(Clone, Copy, Debug, Default)]
pub struct BlockAcctCookie {
    bytes: u64,
    start_time_ns: i64,
    /// `None` is `BLOCK_ACCT_NONE`: the request was accounted already.
    ty: Option<BlockAcctType>,
}

/// `TimedAverageWindow`.
#[derive(Clone, Copy, Debug)]
struct Window {
    min: u64,
    max: u64,
    sum: u64,
    count: u64,
    expiration: i64,
}

impl Window {
    fn reset(&mut self) {
        self.min = u64::MAX;
        self.max = 0;
        self.sum = 0;
        self.count = 0;
    }

    /// `update_expiration()`.
    fn update_expiration(&mut self, now: i64, period: i64) {
        let elapsed = (now - self.expiration) % period;
        self.expiration = now + (period - elapsed);
    }
}

/// `TimedAverage`: min, max and average of the values of about the last period, from two
/// windows half a period apart.
#[derive(Clone, Copy, Debug)]
struct TimedAverage {
    windows: [Window; 2],
    current: usize,
    period: u64,
}

impl TimedAverage {
    /// `timed_average_init()`.
    fn new(now: i64, period: u64) -> Self {
        // Values come from the oldest window, so from [period / 2, period). Asking for 4/3
        // of the period puts them in [2/3 period, 4/3 period).
        let period = period * 4 / 3;
        let mut w = Window { min: u64::MAX, max: 0, sum: 0, count: 0, expiration: 0 };
        w.reset();
        let mut ta = TimedAverage { windows: [w; 2], current: 0, period };
        ta.windows[0].expiration = now + (period / 2) as i64;
        ta.windows[1].expiration = now + period as i64;
        ta
    }

    /// `check_expirations()`: the time elapsed in the current window.
    fn check_expirations(&mut self, now: i64) -> u64 {
        let period = self.period as i64;
        for w in &mut self.windows {
            if w.expiration <= now {
                w.reset();
                w.update_expiration(now, period);
            }
        }
        self.current = if self.windows[0].expiration < self.windows[1].expiration { 0 } else { 1 };
        let remaining = self.windows[self.current].expiration - now;
        (period - remaining) as u64
    }

    /// `timed_average_account()`.
    fn account(&mut self, now: i64, value: u64) {
        self.check_expirations(now);
        for w in &mut self.windows {
            w.sum = w.sum.wrapping_add(value);
            w.count += 1;
            w.min = w.min.min(value);
            w.max = w.max.max(value);
        }
    }

    /// `timed_average_min()`, `_max()` and `_avg()`.
    fn min_max_avg(&mut self, now: i64) -> (u64, u64, u64) {
        self.check_expirations(now);
        let w = &self.windows[self.current];
        let min = if w.min < u64::MAX { w.min } else { 0 };
        let avg = w.sum.checked_div(w.count).unwrap_or(0);
        (min, w.max, avg)
    }

    /// `timed_average_sum()`: the sum and the time elapsed in the window.
    fn sum(&mut self, now: i64) -> (u64, u64) {
        let elapsed = self.check_expirations(now);
        (self.windows[self.current].sum, elapsed)
    }
}

/// `BlockAcctTimedStats`.
#[derive(Clone, Debug)]
struct TimedStats {
    latency: [TimedAverage; MAX_IOTYPE],
    /// In seconds.
    interval_length: u32,
}

/// `BlockLatencyHistogram`: `boundaries` split the latencies into `boundaries.len() + 1`
/// bins, `[0, b0), [b0, b1), ..., [bn, +inf)`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Histogram {
    boundaries: Vec<u64>,
    bins: Vec<u64>,
}

impl Histogram {
    /// `block_latency_histogram_account()`.
    fn account(&mut self, latency_ns: i64) {
        let l = latency_ns.max(0) as u64;
        // The bin is the number of boundaries at or below the latency.
        let i = self.boundaries.partition_point(|&b| b <= l);
        self.bins[i] += 1;
    }
}

/// The counters of `BlockAcctStats`, under its lock.
#[derive(Debug)]
struct Inner {
    nr_bytes: [u64; MAX_IOTYPE],
    nr_ops: [u64; MAX_IOTYPE],
    invalid_ops: [u64; MAX_IOTYPE],
    failed_ops: [u64; MAX_IOTYPE],
    total_time_ns: [u64; MAX_IOTYPE],
    merged: [u64; MAX_IOTYPE],
    last_access_time_ns: i64,
    /// Newest first, as QEMU's list.
    intervals: Vec<TimedStats>,
    account_invalid: bool,
    account_failed: bool,
    latency_histogram: [Option<Histogram>; MAX_IOTYPE],
}

/// `BlockAcctStats`.
#[derive(Debug)]
pub struct BlockAcctStats {
    clock: Arc<dyn Clock>,
    lock: Mutex<Inner>,
}

impl Default for BlockAcctStats {
    fn default() -> Self {
        Self::new(realtime_clock())
    }
}

impl BlockAcctStats {
    /// `block_acct_init()`: invalid and failed requests are accounted.
    pub fn new(clock: Arc<dyn Clock>) -> Self {
        BlockAcctStats {
            clock,
            lock: Mutex::new(Inner {
                nr_bytes: [0; MAX_IOTYPE],
                nr_ops: [0; MAX_IOTYPE],
                invalid_ops: [0; MAX_IOTYPE],
                failed_ops: [0; MAX_IOTYPE],
                total_time_ns: [0; MAX_IOTYPE],
                merged: [0; MAX_IOTYPE],
                last_access_time_ns: 0,
                intervals: Vec::new(),
                account_invalid: true,
                account_failed: true,
                latency_histogram: Default::default(),
            }),
        }
    }

    /// `block_acct_setup()`. `None` for `account_invalid` or `account_failed` is `auto`: the
    /// setting stays as it is.
    pub fn setup(
        &self,
        account_invalid: Option<bool>,
        account_failed: Option<bool>,
        stats_intervals: &[u32],
    ) -> Result<()> {
        {
            let mut s = self.lock.lock().unwrap();
            if let Some(v) = account_invalid {
                s.account_invalid = v;
            }
            if let Some(v) = account_failed {
                s.account_failed = v;
            }
        }
        for &i in stats_intervals {
            if i == 0 {
                return Err(Error::generic(format!("Invalid interval length: {i}")));
            }
            self.add_interval(i);
        }
        Ok(())
    }

    /// `block_acct_add_interval()`: keeps latency averages over `interval_length` seconds.
    pub fn add_interval(&self, interval_length: u32) {
        let now = self.clock.now_ns();
        let period = u64::from(interval_length) * NANOSECONDS_PER_SECOND;
        let ts =
            TimedStats { latency: [TimedAverage::new(now, period); MAX_IOTYPE], interval_length };
        self.lock.lock().unwrap().intervals.insert(0, ts);
    }

    /// `block_acct_start()`.
    pub fn start(&self, bytes: u64, ty: BlockAcctType) -> BlockAcctCookie {
        BlockAcctCookie { bytes, start_time_ns: self.clock.now_ns(), ty: Some(ty) }
    }

    /// `block_account_one_io()`.
    fn account_one_io(&self, cookie: &mut BlockAcctCookie, failed: bool) {
        let time_ns = self.clock.now_ns();
        let latency_ns = time_ns - cookie.start_time_ns;
        let Some(ty) = cookie.ty.take() else {
            return;
        };
        let t = ty as usize;

        let mut s = self.lock.lock().unwrap();
        if failed {
            s.failed_ops[t] += 1;
        } else {
            s.nr_bytes[t] += cookie.bytes;
            s.nr_ops[t] += 1;
        }
        if let Some(h) = &mut s.latency_histogram[t] {
            h.account(latency_ns);
        }
        if !failed || s.account_failed {
            s.total_time_ns[t] = s.total_time_ns[t].wrapping_add(latency_ns as u64);
            s.last_access_time_ns = time_ns;
            for ts in &mut s.intervals {
                ts.latency[t].account(time_ns, latency_ns as u64);
            }
        }
    }

    /// `block_acct_done()`.
    pub fn done(&self, cookie: &mut BlockAcctCookie) {
        self.account_one_io(cookie, false);
    }

    /// `block_acct_failed()`.
    pub fn failed(&self, cookie: &mut BlockAcctCookie) {
        self.account_one_io(cookie, true);
    }

    /// `block_acct_invalid()`: a request refused before any I/O, so it adds no time.
    pub fn invalid(&self, ty: BlockAcctType) {
        let mut s = self.lock.lock().unwrap();
        s.invalid_ops[ty as usize] += 1;
        if s.account_invalid {
            s.last_access_time_ns = self.clock.now_ns();
        }
    }

    /// `block_acct_merge_done()`.
    pub fn merge_done(&self, ty: BlockAcctType, num_requests: u64) {
        self.lock.lock().unwrap().merged[ty as usize] += num_requests;
    }

    /// `block_acct_idle_time_ns()`.
    pub fn idle_time_ns(&self) -> i64 {
        self.clock.now_ns() - self.lock.lock().unwrap().last_access_time_ns
    }

    /// `block_latency_histogram_set()`: `boundaries` must be increasing and not empty.
    /// The error is `-EINVAL`, which the QMP command turns into its own message.
    pub fn latency_histogram_set(&self, ty: BlockAcctType, boundaries: &[u64]) -> io::Result<()> {
        let einval = || io::Error::from(io::ErrorKind::InvalidInput);
        let mut prev = 0;
        for &b in boundaries {
            if b <= prev {
                return Err(einval());
            }
            prev = b;
        }
        if boundaries.is_empty() {
            return Err(einval());
        }
        let h = Histogram { boundaries: boundaries.to_vec(), bins: vec![0; boundaries.len() + 1] };
        self.lock.lock().unwrap().latency_histogram[ty as usize] = Some(h);
        Ok(())
    }

    /// `block_latency_histograms_clear()`.
    pub fn latency_histograms_clear(&self) {
        self.lock.lock().unwrap().latency_histogram = Default::default();
    }

    /// The `BlockDeviceStats` of `query-blockstats`, as `bdrv_query_blk_stats()` fills it.
    pub fn query(&self) -> BlockDeviceStats {
        use BlockAcctType::*;
        let now = self.clock.now_ns();
        let mut s = self.lock.lock().unwrap();
        let n = |v: u64| v as i64;
        let hist = |h: &Option<Histogram>| {
            h.as_ref().map(|h| BlockLatencyHistogramInfo {
                boundaries: h.boundaries.clone(),
                bins: h.bins.clone(),
            })
        };

        // QEMU prepends each interval of its newest-first list, so the oldest comes first.
        let mut timed_stats = Vec::new();
        for ts in s.intervals.iter_mut().rev() {
            let mut lat = |t: BlockAcctType| {
                let (min, max, avg) = ts.latency[t as usize].min_max_avg(now);
                (n(min), n(max), n(avg))
            };
            let (min_rd, max_rd, avg_rd) = lat(Read);
            let (min_wr, max_wr, avg_wr) = lat(Write);
            let (min_za, max_za, avg_za) = lat(ZoneAppend);
            let (min_fl, max_fl, avg_fl) = lat(Flush);
            let mut depth = |t: BlockAcctType| {
                let (sum, elapsed) = ts.latency[t as usize].sum(now);
                sum as f64 / elapsed as f64
            };
            let (rd_depth, wr_depth, za_depth) = (depth(Read), depth(Write), depth(ZoneAppend));
            timed_stats.push(BlockDeviceTimedStats {
                interval_length: i64::from(ts.interval_length),
                min_rd_latency_ns: min_rd,
                max_rd_latency_ns: max_rd,
                avg_rd_latency_ns: avg_rd,
                min_wr_latency_ns: min_wr,
                max_wr_latency_ns: max_wr,
                avg_wr_latency_ns: avg_wr,
                min_zone_append_latency_ns: min_za,
                max_zone_append_latency_ns: max_za,
                avg_zone_append_latency_ns: avg_za,
                min_flush_latency_ns: min_fl,
                max_flush_latency_ns: max_fl,
                avg_flush_latency_ns: avg_fl,
                avg_rd_queue_depth: rd_depth,
                avg_wr_queue_depth: wr_depth,
                avg_zone_append_queue_depth: za_depth,
            });
        }

        let (r, w, z, f, u) =
            (Read as usize, Write as usize, ZoneAppend as usize, Flush as usize, Unmap as usize);
        BlockDeviceStats {
            rd_bytes: n(s.nr_bytes[r]),
            wr_bytes: n(s.nr_bytes[w]),
            zone_append_bytes: n(s.nr_bytes[z]),
            unmap_bytes: n(s.nr_bytes[u]),
            rd_operations: n(s.nr_ops[r]),
            wr_operations: n(s.nr_ops[w]),
            zone_append_operations: n(s.nr_ops[z]),
            flush_operations: n(s.nr_ops[f]),
            unmap_operations: n(s.nr_ops[u]),
            rd_total_time_ns: n(s.total_time_ns[r]),
            wr_total_time_ns: n(s.total_time_ns[w]),
            zone_append_total_time_ns: n(s.total_time_ns[z]),
            flush_total_time_ns: n(s.total_time_ns[f]),
            unmap_total_time_ns: n(s.total_time_ns[u]),
            wr_highest_offset: 0,
            rd_merged: n(s.merged[r]),
            wr_merged: n(s.merged[w]),
            zone_append_merged: n(s.merged[z]),
            unmap_merged: n(s.merged[u]),
            idle_time_ns: (s.last_access_time_ns > 0).then(|| now - s.last_access_time_ns),
            failed_rd_operations: n(s.failed_ops[r]),
            failed_wr_operations: n(s.failed_ops[w]),
            failed_zone_append_operations: n(s.failed_ops[z]),
            failed_flush_operations: n(s.failed_ops[f]),
            failed_unmap_operations: n(s.failed_ops[u]),
            invalid_rd_operations: n(s.invalid_ops[r]),
            invalid_wr_operations: n(s.invalid_ops[w]),
            invalid_zone_append_operations: n(s.invalid_ops[z]),
            invalid_flush_operations: n(s.invalid_ops[f]),
            invalid_unmap_operations: n(s.invalid_ops[u]),
            account_invalid: s.account_invalid,
            account_failed: s.account_failed,
            timed_stats,
            rd_latency_histogram: hist(&s.latency_histogram[r]),
            wr_latency_histogram: hist(&s.latency_histogram[w]),
            zone_append_latency_histogram: hist(&s.latency_histogram[z]),
            flush_latency_histogram: hist(&s.latency_histogram[f]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::throttle::VirtualClock;

    const MS: i64 = 1_000_000;

    fn stats() -> (Arc<VirtualClock>, BlockAcctStats) {
        let clock = Arc::new(VirtualClock::new(1000 * MS));
        (clock.clone(), BlockAcctStats::new(clock))
    }

    #[test]
    fn counters() {
        let (clock, st) = stats();
        let q = st.query();
        assert_eq!(q.idle_time_ns, None);
        assert!(q.account_invalid && q.account_failed);

        let mut c = st.start(4096, BlockAcctType::Read);
        clock.advance(2 * MS);
        st.done(&mut c);
        // A cookie is only accounted once.
        st.done(&mut c);
        let mut c = st.start(512, BlockAcctType::Write);
        clock.advance(MS);
        st.failed(&mut c);
        let mut c = st.start(0, BlockAcctType::Flush);
        st.done(&mut c);
        let mut c = st.start(1 << 20, BlockAcctType::Unmap);
        st.done(&mut c);
        st.invalid(BlockAcctType::Read);
        st.merge_done(BlockAcctType::Write, 3);
        clock.advance(5 * MS);

        let q = st.query();
        assert_eq!((q.rd_bytes, q.rd_operations, q.rd_total_time_ns), (4096, 1, 2 * MS));
        assert_eq!((q.wr_bytes, q.wr_operations, q.failed_wr_operations), (0, 0, 1));
        assert_eq!(q.wr_total_time_ns, MS);
        assert_eq!(q.flush_operations, 1);
        assert_eq!((q.unmap_bytes, q.unmap_operations), (1 << 20, 1));
        assert_eq!(q.invalid_rd_operations, 1);
        assert_eq!(q.wr_merged, 3);
        assert_eq!(q.idle_time_ns, Some(5 * MS));
    }

    #[test]
    fn setup_and_failed_time() {
        let (clock, st) = stats();
        st.setup(Some(false), Some(false), &[]).unwrap();
        let mut c = st.start(512, BlockAcctType::Write);
        clock.advance(MS);
        st.failed(&mut c);
        st.invalid(BlockAcctType::Write);
        let q = st.query();
        assert_eq!((q.failed_wr_operations, q.invalid_wr_operations), (1, 1));
        // Neither adds time nor counts as an access.
        assert_eq!(q.wr_total_time_ns, 0);
        assert_eq!(q.idle_time_ns, None);
        assert!(!q.account_invalid && !q.account_failed);

        // auto keeps the setting.
        st.setup(None, Some(true), &[]).unwrap();
        let q = st.query();
        assert!(!q.account_invalid && q.account_failed);

        assert_eq!(
            st.setup(None, None, &[10, 0]).unwrap_err().message(),
            "Invalid interval length: 0"
        );
    }

    #[test]
    fn intervals() {
        let (clock, st) = stats();
        st.setup(None, None, &[1, 60]).unwrap();
        for l in [MS, 3 * MS] {
            let mut c = st.start(512, BlockAcctType::Read);
            clock.advance(l);
            st.done(&mut c);
        }
        let q = st.query();
        assert_eq!(q.timed_stats.len(), 2);
        assert_eq!(q.timed_stats[0].interval_length, 1);
        assert_eq!(q.timed_stats[1].interval_length, 60);
        let t = &q.timed_stats[0];
        assert_eq!(
            (t.min_rd_latency_ns, t.max_rd_latency_ns, t.avg_rd_latency_ns),
            (MS, 3 * MS, 2 * MS)
        );
        assert_eq!(t.min_wr_latency_ns, 0);
        assert!(t.avg_rd_queue_depth > 0.0);

        // After two periods both windows have been reset.
        clock.advance(3000 * MS);
        let q = st.query();
        assert_eq!(q.timed_stats[0].max_rd_latency_ns, 0);
        assert_eq!(q.timed_stats[1].max_rd_latency_ns, 3 * MS);
    }

    #[test]
    fn histogram() {
        let (clock, st) = stats();
        assert!(st.latency_histogram_set(BlockAcctType::Read, &[]).is_err());
        assert!(st.latency_histogram_set(BlockAcctType::Read, &[10, 10]).is_err());
        st.latency_histogram_set(BlockAcctType::Read, &[10, 50, 100]).unwrap();
        for l in [0, 9, 10, 49, 50, 99, 100, 1000] {
            let mut c = st.start(1, BlockAcctType::Read);
            clock.advance(l);
            st.done(&mut c);
        }
        let h = st.query().rd_latency_histogram.unwrap();
        assert_eq!(h.boundaries, vec![10, 50, 100]);
        assert_eq!(h.bins, vec![2, 2, 2, 2]);
        assert!(st.query().wr_latency_histogram.is_none());
        st.latency_histograms_clear();
        assert!(st.query().rd_latency_histogram.is_none());
    }
}
