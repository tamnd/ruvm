// SPDX-License-Identifier: MIT OR Apache-2.0

//! The event loop: one reactor per thread, with timers on QEMU's four clocks, bottom halves, event
//! notifiers, fd handlers, a thread pool for blocking work and a small executor for futures.
//!
//! This is what QEMU spreads over util/async.c, util/aio-posix.c, util/main-loop.c,
//! util/qemu-timer.c and util/thread-pool.c. A [`Reactor`] belongs to one thread. Other threads
//! reach it through a [`ReactorHandle`], which can schedule bottom halves, post closures and wake it.
//!
//! Readiness comes from mio, which is epoll on Linux, kqueue on macOS and the BSDs, and IOCP on
//! Windows. The io_uring submission path for the block layer sits on top of this in a later
//! milestone and does not change the interface here.

#![forbid(unsafe_code)]

mod bh;
mod clock;
mod pool;
mod reactor;

pub use bh::{Bh, EventNotifier, EventNotifierState};
pub use clock::{Clock, ClockType};
pub use mio::{Interest, Token, event::Source, net};
pub use pool::ThreadPool;
pub use reactor::{
    FdToken, PollTuning, Reactor, ReactorHandle, Ready, TaskId, Timer, current, sleep,
};
