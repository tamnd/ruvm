// SPDX-License-Identifier: MIT OR Apache-2.0

//! A pool of threads for work that blocks, util/thread-pool.c. File IO on hosts without completion
//! based file IO goes here, and so does anything else that would stall a reactor.
//!
//! Threads are started on demand up to a maximum and exit after they have been idle for a while,
//! like QEMU's pool with its `thread-pool-min` and `thread-pool-max` properties.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crate::ReactorHandle;
use crate::reactor::Reactor;

type Job = Box<dyn FnOnce() + Send>;

#[derive(Clone)]
pub struct ThreadPool {
    inner: Arc<PoolInner>,
}

struct PoolInner {
    state: Mutex<PoolState>,
    cond: Condvar,
    idle_timeout: Duration,
}

struct PoolState {
    queue: VecDeque<Job>,
    threads: usize,
    idle: usize,
    min: usize,
    max: usize,
}

impl std::fmt::Debug for ThreadPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self.lock();
        f.debug_struct("ThreadPool")
            .field("threads", &s.threads)
            .field("queued", &s.queue.len())
            .finish()
    }
}

impl ThreadPool {
    /// QEMU's defaults are no minimum and 64 threads at most.
    pub fn new(min: usize, max: usize) -> Self {
        assert!(max >= 1 && min <= max, "a pool needs 1 <= max and min <= max");
        let pool = ThreadPool {
            inner: Arc::new(PoolInner {
                state: Mutex::new(PoolState {
                    queue: VecDeque::new(),
                    threads: 0,
                    idle: 0,
                    min,
                    max,
                }),
                cond: Condvar::new(),
                idle_timeout: Duration::from_secs(10),
            }),
        };
        for _ in 0..min {
            pool.spawn_thread(&mut pool.lock());
        }
        pool
    }

    /// Runs `work` on a pool thread and then `done` with its result on the reactor behind `home`.
    pub fn submit<R: Send + 'static>(
        &self,
        home: &ReactorHandle,
        work: impl FnOnce() -> R + Send + 'static,
        done: impl FnOnce(&mut Reactor, R) + Send + 'static,
    ) {
        let home = home.clone();
        self.execute(move || {
            let r = work();
            home.post(move |reactor| done(reactor, r));
        });
    }

    /// Runs `work` on a pool thread and resolves to its result, for code running on the executor.
    pub fn run<R: Send + 'static>(
        &self,
        work: impl FnOnce() -> R + Send + 'static,
    ) -> impl Future<Output = R> {
        let (tx, rx) = crate::reactor::oneshot();
        self.execute(move || tx.send(work()));
        rx
    }

    /// Changes the bounds, as setting `thread-pool-min` and `thread-pool-max` does.
    pub fn set_bounds(&self, min: usize, max: usize) {
        assert!(max >= 1 && min <= max, "a pool needs 1 <= max and min <= max");
        let mut s = self.lock();
        s.min = min;
        s.max = max;
        while s.threads < min {
            self.spawn_thread(&mut s);
        }
        drop(s);
        self.inner.cond.notify_all();
    }

    pub fn threads(&self) -> usize {
        self.lock().threads
    }

    fn execute(&self, job: impl FnOnce() + Send + 'static) {
        let mut s = self.lock();
        s.queue.push_back(Box::new(job));
        if s.idle == 0 && s.threads < s.max {
            self.spawn_thread(&mut s);
        }
        drop(s);
        self.inner.cond.notify_one();
    }

    fn spawn_thread(&self, s: &mut PoolState) {
        s.threads += 1;
        let inner = Arc::clone(&self.inner);
        let spawned =
            std::thread::Builder::new().name("worker".into()).spawn(move || worker(inner));
        if spawned.is_err() {
            s.threads -= 1;
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, PoolState> {
        self.inner.state.lock().unwrap_or_else(|p| p.into_inner())
    }
}

fn worker(inner: Arc<PoolInner>) {
    let mut s = inner.state.lock().unwrap_or_else(|p| p.into_inner());
    loop {
        if let Some(job) = s.queue.pop_front() {
            drop(s);
            job();
            s = inner.state.lock().unwrap_or_else(|p| p.into_inner());
            continue;
        }
        if s.threads > s.max {
            break;
        }
        s.idle += 1;
        let (next, timeout) =
            inner.cond.wait_timeout(s, inner.idle_timeout).unwrap_or_else(|p| p.into_inner());
        s = next;
        s.idle -= 1;
        if timeout.timed_out() && s.queue.is_empty() && s.threads > s.min {
            break;
        }
    }
    s.threads -= 1;
}
