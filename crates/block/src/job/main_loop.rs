// SPDX-License-Identifier: GPL-2.0-or-later

//! The bits of QEMU's main loop that jobs need: the big QEMU lock and bottom halves.
//!
//! In QEMU the job coroutine runs in an AioContext, and when it returns it schedules
//! `job_exit()` as a bottom half in the main loop, which runs it with the BQL held. The
//! monitor commands that manage jobs run with the BQL held too, so completion and the
//! commands never overlap. A thread that waits in `AIO_WAIT_WHILE()` from the main loop runs
//! pending bottom halves while it waits.
//!
//! Here every job runs on a thread of its own and there is no event loop. [`bql_lock`] is a
//! reentrant lock that stands for the BQL, and [`schedule_bh`] queues a function to run with
//! it held: right away on the scheduling thread if the lock is free, or else by the thread
//! that holds it, either when it lets go of it or while it waits in
//! [`crate::drain::aio_wait_while`], which calls [`poll_bhs`].

use std::cell::Cell;
use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};
use std::thread::ThreadId;

use crate::drain::aio_wait_kick;

struct Bql {
    owner: Option<ThreadId>,
    depth: usize,
}

static BQL: Mutex<Bql> = Mutex::new(Bql { owner: None, depth: 0 });
static BQL_COND: Condvar = Condvar::new();

type Bh = Box<dyn FnOnce() + Send>;

static BHS: Mutex<VecDeque<Bh>> = Mutex::new(VecDeque::new());

thread_local! {
    /// How deep this thread holds the job lock. Bottom halves take the job lock themselves,
    /// so they never run on a thread that holds it.
    static JOB_LOCK_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// Notes that this thread took (`+1`) or dropped (`-1`) the job lock.
pub(crate) fn note_job_lock(delta: isize) {
    JOB_LOCK_DEPTH.with(|d| d.set(d.get().wrapping_add_signed(delta)));
}

fn job_lock_held() -> bool {
    JOB_LOCK_DEPTH.with(Cell::get) > 0
}

/// The BQL, held until dropped. It is reentrant on one thread.
#[must_use]
pub(crate) struct BqlGuard(());

/// `bql_lock()`.
pub(crate) fn bql_lock() -> BqlGuard {
    let me = std::thread::current().id();
    let mut b = BQL.lock().unwrap();
    while b.owner.is_some_and(|t| t != me) {
        b = BQL_COND.wait(b).unwrap();
    }
    b.owner = Some(me);
    b.depth += 1;
    BqlGuard(())
}

/// The BQL if no other thread holds it.
fn bql_try_lock() -> Option<BqlGuard> {
    let me = std::thread::current().id();
    let mut b = BQL.lock().unwrap();
    if b.owner.is_some_and(|t| t != me) {
        return None;
    }
    b.owner = Some(me);
    b.depth += 1;
    Some(BqlGuard(()))
}

/// `bql_locked()`: whether this thread holds the BQL.
pub(crate) fn bql_locked() -> bool {
    BQL.lock().unwrap().owner == Some(std::thread::current().id())
}

impl Drop for BqlGuard {
    fn drop(&mut self) {
        let released = {
            let mut b = BQL.lock().unwrap();
            b.depth -= 1;
            if b.depth == 0 {
                b.owner = None;
                BQL_COND.notify_all();
                true
            } else {
                false
            }
        };
        // A bottom half scheduled while the lock was held runs now.
        if released {
            try_run_bhs();
        }
    }
}

/// `aio_bh_schedule_oneshot(qemu_get_aio_context(), ...)`.
pub(crate) fn schedule_bh(f: impl FnOnce() + Send + 'static) {
    BHS.lock().unwrap().push_back(Box::new(f));
    aio_wait_kick();
    try_run_bhs();
}

/// Runs the pending bottom halves if the BQL is free, or held by this thread at its outer
/// level only because of this call.
fn try_run_bhs() {
    if job_lock_held() || bql_locked() {
        return;
    }
    if BHS.lock().unwrap().is_empty() {
        return;
    }
    if let Some(g) = bql_try_lock() {
        run_bhs();
        drop(g);
    }
}

fn run_bhs() {
    loop {
        let Some(bh) = BHS.lock().unwrap().pop_front() else {
            return;
        };
        bh();
    }
}

/// What `AIO_WAIT_WHILE()` in the main loop does on each round: run the pending bottom
/// halves, if this thread holds the BQL. Other threads run them through [`schedule_bh`] or
/// when they let go of the lock.
pub(crate) fn poll_bhs() {
    if job_lock_held() || !bql_locked() {
        return;
    }
    run_bhs();
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::*;

    #[test]
    fn bh_runs_on_release_and_in_wait() {
        let n = Arc::new(AtomicU32::new(0));
        let g = bql_lock();
        let g2 = bql_lock();
        let n2 = n.clone();
        std::thread::spawn(move || schedule_bh(move || _ = n2.fetch_add(1, Ordering::SeqCst)))
            .join()
            .unwrap();
        // This thread holds the BQL, so the other one could not run it.
        assert_eq!(n.load(Ordering::SeqCst), 0);
        crate::drain::aio_wait_while(|| n.load(Ordering::SeqCst) == 0);
        assert_eq!(n.load(Ordering::SeqCst), 1);
        let n3 = n.clone();
        std::thread::spawn(move || schedule_bh(move || _ = n3.fetch_add(1, Ordering::SeqCst)))
            .join()
            .unwrap();
        drop(g2);
        assert_eq!(n.load(Ordering::SeqCst), 1);
        drop(g);
        assert_eq!(n.load(Ordering::SeqCst), 2);
    }
}
