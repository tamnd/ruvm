// SPDX-License-Identifier: GPL-2.0-or-later

//! The graph lock from block/graph-lock.c: requests hold it for reading while they walk the
//! graph, and changes to the graph take it for writing.
//!
//! QEMU keeps a reader count per AioContext and makes the writer poll until every count is
//! zero. Requests here run on plain threads, so this is a reader/writer lock with a reader
//! count per thread instead. Both sides nest on one thread: a reader may take the lock for
//! reading again, the writer may take it for reading or writing again, and a thread that
//! holds it for reading may take it for writing once no other thread reads, as QEMU's main
//! loop does when it changes the graph from inside a drained section.

use std::cell::Cell;
use std::sync::{Condvar, Mutex};
use std::thread::ThreadId;

struct State {
    /// How many threads hold the lock for reading.
    readers: usize,
    /// The thread that holds the lock for writing, and how often.
    writer: Option<(ThreadId, usize)>,
}

static LOCK: Mutex<State> = Mutex::new(State { readers: 0, writer: None });
static COND: Condvar = Condvar::new();

thread_local! {
    /// How often this thread holds the lock for reading.
    static DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// A read lock on the graph, `GRAPH_RDLOCK_GUARD()`.
#[must_use]
pub(crate) struct ReadGuard(());

/// A write lock on the graph, `bdrv_graph_wrlock()` until `bdrv_graph_wrunlock()`.
#[must_use]
pub(crate) struct WriteGuard(());

/// `bdrv_graph_co_rdlock()`.
pub(crate) fn rdlock() -> ReadGuard {
    let depth = DEPTH.with(Cell::get);
    if depth == 0 {
        let me = std::thread::current().id();
        let mut s = LOCK.lock().unwrap();
        while s.writer.is_some_and(|(t, _)| t != me) {
            s = COND.wait(s).unwrap();
        }
        s.readers += 1;
    }
    DEPTH.with(|d| d.set(depth + 1));
    ReadGuard(())
}

impl Drop for ReadGuard {
    fn drop(&mut self) {
        let depth = DEPTH.with(Cell::get);
        DEPTH.with(|d| d.set(depth - 1));
        if depth == 1 {
            let mut s = LOCK.lock().unwrap();
            s.readers -= 1;
            drop(s);
            COND.notify_all();
        }
    }
}

/// `bdrv_graph_wrlock()`: waits until no other thread reads or writes.
pub(crate) fn wrlock() -> WriteGuard {
    let me = std::thread::current().id();
    let mine = usize::from(DEPTH.with(Cell::get) > 0);
    let mut s = LOCK.lock().unwrap();
    loop {
        match s.writer {
            Some((t, n)) if t == me => {
                s.writer = Some((t, n + 1));
                return WriteGuard(());
            }
            None if s.readers == mine => {
                s.writer = Some((me, 1));
                return WriteGuard(());
            }
            _ => s = COND.wait(s).unwrap(),
        }
    }
}

impl Drop for WriteGuard {
    fn drop(&mut self) {
        let mut s = LOCK.lock().unwrap();
        match s.writer {
            Some((t, n)) if n > 1 => s.writer = Some((t, n - 1)),
            _ => s.writer = None,
        }
        drop(s);
        COND.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn nesting_and_exclusion() {
        let r1 = rdlock();
        let r2 = rdlock();
        // This thread is the only reader, so it may write.
        let w = wrlock();
        let r3 = rdlock();
        drop((r3, w, r2, r1));

        let r = rdlock();
        let done = Arc::new(AtomicBool::new(false));
        let d = done.clone();
        let t = std::thread::spawn(move || {
            let _w = wrlock();
            d.store(true, Ordering::SeqCst);
        });
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert!(!done.load(Ordering::SeqCst), "the writer waits for the reader");
        drop(r);
        t.join().unwrap();
        assert!(done.load(Ordering::SeqCst));
    }
}
