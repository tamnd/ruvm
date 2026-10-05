// SPDX-License-Identifier: GPL-2.0-or-later

//! Device timers, `QEMUTimer` and the clocks behind it from util/qemu-timer.c.
//!
//! A [`Clock`] owns the timer list of one clock type. Its time comes from a [`TimeSource`]:
//! under qtest the virtual clock only moves when the test steps it, otherwise it follows the
//! host. Callbacks run with no lock held, so they can read the clock and rearm any timer,
//! including their own.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use ruvm_base::{ClockType, TimerId, TimerList};

/// `NANOSECONDS_PER_SECOND`.
pub const NANOSECONDS_PER_SECOND: i64 = 1_000_000_000;

/// Where a clock reads its time.
#[derive(Debug)]
pub enum TimeSource {
    /// Only moves when [`Clock::advance_to`] is called. This is the virtual clock under qtest.
    Manual,
    /// Nanoseconds since `start`, like `get_clock()` for the realtime clock.
    Monotonic(Instant),
    /// The host wall clock in nanoseconds since the epoch, `get_clock_realtime()`.
    Wall,
}

type Callback = Arc<dyn Fn() + Send + Sync>;

struct Inner {
    now: i64,
    list: TimerList<Callback>,
}

type Notify = Arc<dyn Fn() + Send + Sync>;

/// One clock and its timer list, `QEMUClock` plus its main loop `QEMUTimerList`.
pub struct Clock {
    kind: ClockType,
    source: TimeSource,
    inner: Mutex<Inner>,
    notify: Mutex<Option<Notify>>,
}

impl fmt::Debug for Clock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Clock").field("kind", &self.kind).field("source", &self.source).finish()
    }
}

impl Clock {
    pub fn new(kind: ClockType, source: TimeSource) -> Arc<Self> {
        Arc::new(Clock {
            kind,
            source,
            inner: Mutex::new(Inner { now: 0, list: TimerList::new() }),
            notify: Mutex::new(None),
        })
    }

    /// A virtual clock that only moves when stepped, what qtest runs on.
    pub fn manual(kind: ClockType) -> Arc<Self> {
        Self::new(kind, TimeSource::Manual)
    }

    pub fn kind(&self) -> ClockType {
        self.kind
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn read(&self, inner: &Inner) -> i64 {
        match self.source {
            TimeSource::Manual => inner.now,
            TimeSource::Monotonic(start) => start.elapsed().as_nanos() as i64,
            TimeSource::Wall => {
                SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos() as i64)
            }
        }
    }

    /// `qemu_clock_get_ns()`.
    pub fn get_ns(&self) -> i64 {
        let inner = self.lock();
        self.read(&inner)
    }

    /// `qemu_clock_get_ms()`.
    pub fn get_ms(&self) -> i64 {
        self.get_ns() / 1_000_000
    }

    /// `timer_new_ns()`. The callback runs on whatever thread advances or polls the clock.
    pub fn new_timer(self: &Arc<Self>, callback: impl Fn() + Send + Sync + 'static) -> Timer {
        let id = self.lock().list.insert(Arc::new(callback));
        Timer { clock: Arc::downgrade(self), id }
    }

    /// Sets the `notify_cb` of the timer list: called, with no lock held, when arming a timer
    /// makes it the earliest one, so that whatever waits for the next deadline can wait again.
    pub fn set_notify(&self, notify: impl Fn() + Send + Sync + 'static) {
        *self.notify.lock().unwrap_or_else(|p| p.into_inner()) = Some(Arc::new(notify));
    }

    /// `timerlist_notify()`.
    fn notify(&self) {
        let n = self.notify.lock().unwrap_or_else(|p| p.into_inner()).clone();
        if let Some(n) = n {
            n();
        }
    }

    /// The earliest armed deadline, if any.
    pub fn next_deadline(&self) -> Option<i64> {
        self.lock().list.next_deadline()
    }

    /// `qemu_clock_deadline_ns_all()`: nanoseconds to the next timer, 0 if one is overdue, or
    /// -1 when none is armed.
    pub fn deadline_ns(&self) -> i64 {
        let mut inner = self.lock();
        let now = self.read(&inner);
        match inner.list.next_deadline() {
            Some(d) => (d - now).max(0),
            None => -1,
        }
    }

    /// `qemu_clock_run_timers()`: fires every timer due at the current time. Returns how many
    /// ran.
    pub fn run_timers(&self) -> usize {
        let mut fired = 0;
        loop {
            // Take one due timer at a time, so a callback that rearms or deletes other timers is
            // seen before the next one fires.
            let cb = {
                let mut inner = self.lock();
                let now = self.read(&inner);
                inner.list.pop_expired(now).and_then(|id| inner.list.callback(id).cloned())
            };
            match cb {
                Some(cb) => {
                    cb();
                    fired += 1;
                }
                None => return fired,
            }
        }
    }

    /// `qemu_clock_advance_virtual_time()`: moves a manual clock to `dest`, stopping at every
    /// deadline on the way to run the timers due there. Returns the clock afterwards, which
    /// never goes back.
    pub fn advance_to(&self, dest: i64) -> i64 {
        assert!(matches!(self.source, TimeSource::Manual), "only a manual clock can be stepped");
        loop {
            {
                let mut inner = self.lock();
                let next = inner.list.next_deadline().filter(|&d| d <= dest);
                let Some(d) = next else {
                    inner.now = inner.now.max(dest);
                    break;
                };
                inner.now = inner.now.max(d);
            }
            self.run_timers();
        }
        self.run_timers();
        self.get_ns()
    }
}

/// `QEMUTimer`. Dropping it deletes the timer.
pub struct Timer {
    clock: Weak<Clock>,
    id: TimerId,
}

impl fmt::Debug for Timer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Timer").field("id", &self.id).field("expire", &self.expire_time()).finish()
    }
}

impl Timer {
    fn with<R>(&self, f: impl FnOnce(&mut TimerList<Callback>) -> R) -> Option<R> {
        let clock = self.clock.upgrade()?;
        let mut inner = clock.lock();
        Some(f(&mut inner.list))
    }

    /// `timer_mod_ns()`: arms the timer for `deadline` on its clock.
    pub fn modify(&self, deadline: i64) {
        self.rearm(|l| l.arm(self.id, deadline));
    }

    /// `timer_mod_anticipate_ns()`.
    pub fn modify_anticipate(&self, deadline: i64) {
        self.rearm(|l| l.arm_earlier(self.id, deadline));
    }

    /// Arms the timer with `arm`, then `timerlist_rearm()`: notifies the clock when the timer
    /// is now the first to fire.
    fn rearm(&self, arm: impl FnOnce(&mut TimerList<Callback>)) {
        let Some(clock) = self.clock.upgrade() else { return };
        let first = {
            let mut inner = clock.lock();
            arm(&mut inner.list);
            let d = inner.list.deadline_of(self.id);
            d.is_some() && inner.list.next_deadline() == d
        };
        if first {
            clock.notify();
        }
    }

    /// `timer_del()`.
    pub fn del(&self) {
        self.with(|l| l.disarm(self.id));
    }

    /// `timer_pending()`.
    pub fn pending(&self) -> bool {
        self.with(|l| l.is_armed(self.id)).unwrap_or(false)
    }

    /// `timer_expire_time_ns()`, or `None` when the timer is not armed.
    pub fn expire_time(&self) -> Option<i64> {
        self.with(|l| l.deadline_of(self.id)).flatten()
    }

    /// The clock the timer runs on, gone once the clock has been dropped.
    pub fn clock(&self) -> Option<Arc<Clock>> {
        self.clock.upgrade()
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        self.with(|l| l.remove(self.id));
    }
}

/// `muldiv64()`: `a * b / c` with a 128 bit intermediate.
pub fn muldiv64(a: u64, b: u32, c: u32) -> u64 {
    (u128::from(a) * u128::from(b) / u128::from(c)) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn stepping_runs_timers_at_their_deadlines() {
        let clock = Clock::manual(ClockType::Virtual);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (c, s) = (Arc::downgrade(&clock), seen.clone());
        let t = clock.new_timer(move || s.lock().unwrap().push(c.upgrade().unwrap().get_ns()));
        t.modify(100);
        assert_eq!(clock.deadline_ns(), 100);
        assert_eq!(clock.advance_to(50), 50);
        assert!(seen.lock().unwrap().is_empty());
        assert_eq!(clock.advance_to(250), 250);
        assert_eq!(*seen.lock().unwrap(), [100]);
        assert!(!t.pending());
        assert_eq!(clock.deadline_ns(), -1);
    }

    #[test]
    fn periodic_timer_rearms_itself() {
        let clock = Clock::manual(ClockType::Virtual);
        let count = Arc::new(AtomicUsize::new(0));
        let slot: Arc<Mutex<Option<Timer>>> = Arc::default();
        let (c, n, s) = (Arc::downgrade(&clock), count.clone(), slot.clone());
        let t = clock.new_timer(move || {
            n.fetch_add(1, Ordering::SeqCst);
            let now = c.upgrade().unwrap().get_ns();
            s.lock().unwrap().as_ref().unwrap().modify(now + 10);
        });
        t.modify(10);
        *slot.lock().unwrap() = Some(t);
        clock.advance_to(95);
        assert_eq!(count.load(Ordering::SeqCst), 9);
        assert_eq!(slot.lock().unwrap().as_ref().unwrap().expire_time(), Some(100));
    }

    #[test]
    fn arming_the_first_timer_notifies() {
        let clock = Clock::manual(ClockType::Virtual);
        let count = Arc::new(AtomicUsize::new(0));
        let n = count.clone();
        clock.set_notify(move || {
            n.fetch_add(1, Ordering::SeqCst);
        });
        let (a, b) = (clock.new_timer(|| {}), clock.new_timer(|| {}));
        a.modify(100);
        assert_eq!(count.load(Ordering::SeqCst), 1);
        // Later than the first: nothing waiting for the deadline needs to know.
        b.modify(200);
        assert_eq!(count.load(Ordering::SeqCst), 1);
        b.modify_anticipate(50);
        assert_eq!(count.load(Ordering::SeqCst), 2);
        a.del();
        assert_eq!(count.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn dropped_timers_never_fire() {
        let clock = Clock::manual(ClockType::Virtual);
        let count = Arc::new(AtomicUsize::new(0));
        let n = count.clone();
        let t = clock.new_timer(move || {
            n.fetch_add(1, Ordering::SeqCst);
        });
        t.modify(5);
        drop(t);
        clock.advance_to(10);
        assert_eq!(count.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn muldiv() {
        assert_eq!(muldiv64(u64::MAX, 2, 4), u64::MAX / 2);
        assert_eq!(muldiv64(1_000_000_000, 32768, 1_000_000_000), 32768);
    }
}
