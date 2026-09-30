// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bottom halves and event notifiers.

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use crate::reactor::Shared;

const SCHEDULED: u8 = 1;
const DELETED: u8 = 2;

/// A callback that runs on its reactor soon after it is scheduled, `QEMUBH` from util/async.c.
///
/// Scheduling is cheap and can be done from any thread. Scheduling a bottom half that is already
/// scheduled does nothing, so it runs once however many times it was scheduled before it got to
/// run, which is the property devices rely on to coalesce work.
#[derive(Clone)]
pub struct Bh {
    pub(crate) inner: Arc<BhInner>,
}

pub(crate) struct BhInner {
    flags: AtomicU8,
    callback: Mutex<Box<dyn FnMut() + Send>>,
    shared: Arc<Shared>,
}

impl std::fmt::Debug for Bh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bh").field("flags", &self.inner.flags.load(Ordering::Relaxed)).finish()
    }
}

impl Bh {
    pub(crate) fn new(shared: Arc<Shared>, f: impl FnMut() + Send + 'static) -> Self {
        Bh {
            inner: Arc::new(BhInner {
                flags: AtomicU8::new(0),
                callback: Mutex::new(Box::new(f)),
                shared,
            }),
        }
    }

    /// `qemu_bh_schedule()`.
    pub fn schedule(&self) {
        let old = self.inner.flags.fetch_or(SCHEDULED, Ordering::AcqRel);
        if old & (SCHEDULED | DELETED) == 0 {
            self.inner.shared.push_bh(Arc::clone(&self.inner));
        }
    }

    /// `qemu_bh_cancel()`. A bottom half that is already running finishes.
    pub fn cancel(&self) {
        self.inner.flags.fetch_and(!SCHEDULED, Ordering::AcqRel);
    }

    /// `qemu_bh_delete()`. The bottom half never runs again, even if it is scheduled.
    pub fn delete(&self) {
        self.inner.flags.fetch_or(DELETED, Ordering::AcqRel);
    }

    pub fn is_scheduled(&self) -> bool {
        self.inner.flags.load(Ordering::Acquire) & SCHEDULED != 0
    }
}

impl BhInner {
    /// Runs the callback if the bottom half is still scheduled. Returns whether it ran.
    pub(crate) fn run(&self) -> bool {
        let old = self.flags.fetch_and(!SCHEDULED, Ordering::AcqRel);
        if old & SCHEDULED == 0 || old & DELETED != 0 {
            return false;
        }
        let mut cb = self.callback.lock().unwrap_or_else(|p| p.into_inner());
        cb();
        true
    }
}

/// A flag that any thread can set and that runs a handler on the reactor when it goes from clear to
/// set. This is what QEMU uses an `EventNotifier` with `aio_set_event_notifier()` for. On the host
/// side there is no eventfd, the reactor's own waker does the job.
#[derive(Clone, Debug)]
pub struct EventNotifier {
    flag: Arc<AtomicBool>,
    bh: Bh,
}

impl EventNotifier {
    pub(crate) fn new(
        shared: Arc<Shared>,
        mut handler: impl FnMut(&EventNotifierState) + Send + 'static,
    ) -> Self {
        let flag = Arc::new(AtomicBool::new(false));
        let state = EventNotifierState { flag: Arc::clone(&flag) };
        let bh = Bh::new(shared, move || handler(&state));
        EventNotifier { flag, bh }
    }

    /// `event_notifier_set()`.
    pub fn set(&self) {
        if !self.flag.swap(true, Ordering::AcqRel) {
            self.bh.schedule();
        }
    }

    pub fn is_set(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }

    /// Stops the handler from running again.
    pub fn delete(&self) {
        self.bh.delete();
    }
}

/// What the handler of an [`EventNotifier`] gets, so it can clear the flag the way handlers call
/// `event_notifier_test_and_clear()`.
#[derive(Debug)]
pub struct EventNotifierState {
    flag: Arc<AtomicBool>,
}

impl EventNotifierState {
    pub fn test_and_clear(&self) -> bool {
        self.flag.swap(false, Ordering::AcqRel)
    }
}
