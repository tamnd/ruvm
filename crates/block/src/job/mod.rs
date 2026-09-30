// SPDX-License-Identifier: GPL-2.0-or-later

//! Block jobs: job.c, blockjob.c, job-qmp.c and the jobs themselves (stream, commit, mirror,
//! backup, create and amend).
//!
//! # Differences from QEMU
//!
//! - A job's coroutine is a thread of its own. `job_yield()` and `job_sleep_ns()` wait on a
//!   condition variable, and `job_enter()` only wakes the thread: it does not run the job up
//!   to its next yield before returning, so a caller that wants to see the effect of entering
//!   a job waits for it (the tests poll the job state).
//! - There is no main loop. [`main_loop`] has a reentrant lock that stands for the BQL and a
//!   queue of bottom halves. `job_exit()` runs as one, with the BQL held, either on the job's
//!   thread when nobody holds the BQL, or on the thread that holds it when it waits in
//!   `AIO_WAIT_WHILE()` or lets go of the lock.
//! - Job drivers get installed after `job_create()`/`block_job_create()` returns
//!   ([`core::Job::set_driver`]), because the driver state of a Rust job is built from the
//!   job, where QEMU embeds the job in the driver state.
//! - Operation blockers and frozen backing links live in side tables ([`blocker`]); only the
//!   job paths and `bdrv_replace_node()` check them. Resize, snapshots, reopen and
//!   `blockdev-del` do not.
//! - Jobs do their I/O one chunk at a time on their thread. Mirror has no parallel in-flight
//!   operations and block-copy no parallel tasks, so `max-workers` and the mirror's
//!   `buf-size` only bound the chunk sizes.
//! - AioContexts and iothreads do not exist, so jobs never move between contexts.

pub(crate) mod backup;
pub(crate) mod block_job;
pub(crate) mod blocker;
pub(crate) mod chain;
pub(crate) mod commit;
pub(crate) mod core;
pub(crate) mod create;
pub(crate) mod main_loop;
pub(crate) mod mirror;
pub(crate) mod qmp;
pub(crate) mod ratelimit;
pub(crate) mod stream;

#[cfg(test)]
mod tests;

use crate::event::BlockEvent;

/// Sends a job event, recording it for the tests too.
pub(crate) fn send_event(ev: BlockEvent) {
    #[cfg(test)]
    tests::record_event(&ev);
    crate::event::emit(ev);
}
