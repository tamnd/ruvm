// SPDX-License-Identifier: MIT OR Apache-2.0

//! QEMU's clocks. Every reactor in a process shares one [`Clock`], so stopping the VM stops virtual
//! time everywhere at once.

use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub use ruvm_base::ClockType;

use crate::ReactorHandle;

/// The clock source shared by all reactors.
///
/// Realtime is nanoseconds of monotonic host time. Virtual time runs from zero at startup, stops
/// while the VM is stopped and picks up where it left off when it runs again, which is what
/// `cpu_get_clock()` and `cpus_disable_ticks()` do in QEMU. Host is wall clock nanoseconds since
/// the Unix epoch. There is no icount yet, so virtual realtime equals virtual.
#[derive(Clone)]
pub struct Clock {
    inner: Arc<Inner>,
}

struct Inner {
    start: Instant,
    // Virtual time is realtime plus this offset while running.
    offset: AtomicI64,
    // The frozen value while stopped.
    frozen: AtomicI64,
    running: AtomicBool,
    // Serializes start and stop so the offset and the frozen value move together.
    transition: Mutex<()>,
    reactors: Mutex<Vec<ReactorHandle>>,
}

impl Default for Clock {
    fn default() -> Self {
        Clock::new()
    }
}

impl std::fmt::Debug for Clock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Clock")
            .field("virtual", &self.now(ClockType::Virtual))
            .field("running", &self.is_running())
            .finish()
    }
}

impl Clock {
    /// A clock with the VM stopped, as it is before `vm_start()`.
    pub fn new() -> Self {
        Clock {
            inner: Arc::new(Inner {
                start: Instant::now(),
                offset: AtomicI64::new(0),
                frozen: AtomicI64::new(0),
                running: AtomicBool::new(false),
                transition: Mutex::new(()),
                reactors: Mutex::new(Vec::new()),
            }),
        }
    }

    pub fn now(&self, clock: ClockType) -> i64 {
        match clock {
            ClockType::Realtime => self.realtime(),
            ClockType::Host => match SystemTime::now().duration_since(UNIX_EPOCH) {
                Ok(d) => d.as_nanos() as i64,
                Err(e) => -(e.duration().as_nanos() as i64),
            },
            ClockType::Virtual | ClockType::VirtualRt => {
                if self.inner.running.load(Ordering::Acquire) {
                    self.realtime() + self.inner.offset.load(Ordering::Acquire)
                } else {
                    self.inner.frozen.load(Ordering::Acquire)
                }
            }
        }
    }

    pub fn is_running(&self) -> bool {
        self.inner.running.load(Ordering::Acquire)
    }

    /// Lets virtual time run, `cpus_enable_ticks()`.
    pub fn start(&self) {
        let _g = self.inner.transition.lock().unwrap_or_else(|p| p.into_inner());
        if self.is_running() {
            return;
        }
        let frozen = self.inner.frozen.load(Ordering::Acquire);
        self.inner.offset.store(frozen - self.realtime(), Ordering::Release);
        self.inner.running.store(true, Ordering::Release);
        self.notify();
    }

    /// Stops virtual time, `cpus_disable_ticks()`.
    pub fn stop(&self) {
        let _g = self.inner.transition.lock().unwrap_or_else(|p| p.into_inner());
        if !self.is_running() {
            return;
        }
        let now = self.realtime() + self.inner.offset.load(Ordering::Acquire);
        self.inner.frozen.store(now, Ordering::Release);
        self.inner.running.store(false, Ordering::Release);
        self.notify();
    }

    /// Sets virtual time, which is what an incoming migration does with the source's clock.
    pub fn set_virtual(&self, ns: i64) {
        let _g = self.inner.transition.lock().unwrap_or_else(|p| p.into_inner());
        if self.is_running() {
            self.inner.offset.store(ns - self.realtime(), Ordering::Release);
        } else {
            self.inner.frozen.store(ns, Ordering::Release);
        }
        self.notify();
    }

    pub(crate) fn attach(&self, handle: ReactorHandle) {
        self.inner.reactors.lock().unwrap_or_else(|p| p.into_inner()).push(handle);
    }

    pub(crate) fn detach(&self, handle: &ReactorHandle) {
        self.inner.reactors.lock().unwrap_or_else(|p| p.into_inner()).retain(|h| !h.same(handle));
    }

    // Every reactor with virtual timers has to recompute its deadline, as `qemu_clock_notify()`
    // makes them do.
    fn notify(&self) {
        for h in self.inner.reactors.lock().unwrap_or_else(|p| p.into_inner()).iter() {
            h.wake();
        }
    }

    fn realtime(&self) -> i64 {
        self.inner.start.elapsed().as_nanos() as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn virtual_time_stops_with_the_vm() {
        let c = Clock::new();
        assert_eq!(c.now(ClockType::Virtual), 0);
        c.start();
        std::thread::sleep(std::time::Duration::from_millis(5));
        c.stop();
        let t = c.now(ClockType::Virtual);
        assert!(t >= 5_000_000);
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert_eq!(c.now(ClockType::Virtual), t);
        c.start();
        assert!(c.now(ClockType::Virtual) >= t);
        assert!(c.now(ClockType::Virtual) < t + 5_000_000);
        c.set_virtual(1_000_000_000_000);
        assert!(c.now(ClockType::Virtual) >= 1_000_000_000_000);
    }

    #[test]
    fn host_time_is_wall_clock() {
        let c = Clock::new();
        assert!(c.now(ClockType::Host) > 1_700_000_000_000_000_000);
    }
}
