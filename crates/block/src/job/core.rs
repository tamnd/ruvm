// SPDX-License-Identifier: GPL-2.0-or-later

//! The generic job machinery of job.c: the job list and lock, the state machine, pausing,
//! cancelling, completion, transactions and finalisation.
//!
//! Functions whose QEMU name ends in `_locked` take a [`JobLock`], the `job_mutex`. As in
//! QEMU, driver callbacks run with it released, see [`JobLock::unlocked`]. The functions that
//! wait for a job to finish must also be called with the BQL held (taken before the job
//! lock), so that the `job_exit()` bottom half runs while they wait. See the module
//! documentation of [`crate::job`] for how the coroutine becomes a thread.

use std::any::Any;
use std::io;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, Weak};
use std::time::{Duration, Instant};

use ruvm_base::{Error, Result};
use ruvm_qapi::types::{BlockJobChangeOptions, BlockJobInfo, JobInfo, JobStatus, JobType, JobVerb};

use super::block_job::BlockJob;
use super::main_loop::{note_job_lock, schedule_bh};
use super::ratelimit::Progress;
use crate::drain::{aio_wait_kick, aio_wait_while};
use crate::event::BlockEvent;

/// `JOB_DEFAULT`.
pub(crate) const JOB_DEFAULT: u32 = 0x00;
/// `JOB_INTERNAL`: a job without an id that the monitor does not see.
pub(crate) const JOB_INTERNAL: u32 = 0x01;
/// `JOB_MANUAL_FINALIZE`: wait in PENDING for `job-finalize`.
pub(crate) const JOB_MANUAL_FINALIZE: u32 = 0x02;
/// `JOB_MANUAL_DISMISS`: wait in CONCLUDED for `job-dismiss`.
pub(crate) const JOB_MANUAL_DISMISS: u32 = 0x04;

/// `ECANCELED`.
pub(crate) const ECANCELED: i32 = libc::ECANCELED;

const T: bool = true;
const F: bool = false;

/// `JobSTT`: which state may follow which. Rows and columns are in [`JobStatus`] order:
/// undefined, created, running, paused, ready, standby, waiting, pending, aborting,
/// concluded, null.
const JOB_STT: [[bool; 11]; 11] = [
    /* U: */ [F, T, F, F, F, F, F, F, F, F, F],
    /* C: */ [F, F, T, F, F, F, F, F, T, F, T],
    /* R: */ [F, F, F, T, T, F, T, F, T, F, F],
    /* P: */ [F, F, T, F, F, F, F, F, F, F, F],
    /* Y: */ [F, F, F, F, F, T, T, F, T, F, F],
    /* S: */ [F, F, F, F, T, F, F, F, F, F, F],
    /* W: */ [F, F, F, F, F, F, F, T, T, F, F],
    /* D: */ [F, F, F, F, F, F, F, F, T, T, F],
    /* X: */ [F, F, F, F, F, F, F, F, T, T, F],
    /* E: */ [F, F, F, F, F, F, F, F, F, F, T],
    /* N: */ [F, F, F, F, F, F, F, F, F, F, F],
];

/// `JobVerbTable`: which verb each state accepts, rows in [`JobVerb`] order (cancel, pause,
/// resume, set-speed, complete, dismiss, finalize, change).
const JOB_VERB_TABLE: [[bool; 11]; 8] = [
    /* cancel */ [F, T, T, T, T, T, T, T, F, F, F],
    /* pause */ [F, T, T, T, T, T, F, F, F, F, F],
    /* resume */ [F, T, T, T, T, T, F, F, F, F, F],
    /* set-speed */ [F, T, T, T, T, T, F, F, F, F, F],
    /* complete */ [F, F, F, F, T, T, F, F, F, F, F],
    /* dismiss */ [F, F, F, F, F, F, F, F, F, T, F],
    /* finalize */ [F, F, F, F, F, F, F, T, F, F, F],
    /* change */ [F, T, T, T, T, T, F, F, F, F, F],
];

/// The error of a job's `run` or `prepare`: the negative errno QEMU returns, as a positive
/// number, and the `Error` it may have set.
#[derive(Debug)]
pub(crate) struct JobErr {
    pub errno: i32,
    pub err: Option<Error>,
}

impl JobErr {
    /// Just an errno; the job's error message becomes `strerror()` of it.
    pub(crate) fn errno(errno: i32) -> Self {
        JobErr { errno, err: None }
    }

    /// An errno with a message.
    pub(crate) fn with(errno: i32, err: Error) -> Self {
        JobErr { errno, err: Some(err) }
    }
}

impl From<io::Error> for JobErr {
    fn from(e: io::Error) -> Self {
        JobErr::errno(e.raw_os_error().unwrap_or(libc::EIO))
    }
}

/// `strerror()` of a positive errno.
pub(crate) fn strerror(errno: i32) -> String {
    ruvm_base::error::strerror(&io::Error::from_raw_os_error(errno))
}

/// `BlockCompletionFunc`: called with the job's return value once it is finalised.
pub(crate) type JobCb = Box<dyn FnOnce(i32) + Send>;

/// `JobDriver` with the `BlockJobDriver` additions. Every callback but `run` has a default
/// that behaves like a NULL pointer in QEMU.
pub(crate) trait JobDriver: Send + Sync + 'static {
    /// `.run`: the job proper, on the job's own thread.
    fn run(&self, job: &Arc<Job>) -> Result<(), JobErr>;

    /// `.pause`: called at a pause point before the job pauses.
    fn pause(&self, job: &Arc<Job>) {
        let _ = job;
    }

    /// `.resume`: called at a pause point after the job resumed.
    fn resume(&self, job: &Arc<Job>) {
        let _ = job;
    }

    /// `.user_resume`, besides the iostatus reset every block job does.
    fn user_resume(&self, job: &Arc<Job>) {
        let _ = job;
    }

    /// Whether there is a `.complete`.
    fn has_complete(&self) -> bool {
        false
    }

    /// `.complete`.
    fn complete(&self, job: &Arc<Job>) -> Result<()> {
        let _ = job;
        Ok(())
    }

    /// `.prepare`. `None` is a NULL callback.
    fn prepare(&self, job: &Arc<Job>) -> Option<Result<(), JobErr>> {
        let _ = job;
        None
    }

    /// `.commit`.
    fn commit(&self, job: &Arc<Job>) {
        let _ = job;
    }

    /// `.abort`.
    fn abort(&self, job: &Arc<Job>) {
        let _ = job;
    }

    /// `.clean`.
    fn clean(&self, job: &Arc<Job>) {
        let _ = job;
    }

    /// `.cancel`: returns whether the cancellation is forced after all. `None` is a NULL
    /// callback, which makes every cancellation a forced one.
    fn cancel(&self, job: &Arc<Job>, force: bool) -> Option<bool> {
        let _ = (job, force);
        None
    }

    /// `BlockJobDriver.drained_poll`.
    fn drained_poll(&self, job: &Arc<Job>) -> bool {
        let _ = job;
        true
    }

    /// `BlockJobDriver.set_speed`.
    fn set_speed(&self, job: &Arc<Job>, speed: i64) {
        let _ = (job, speed);
    }

    /// `BlockJobDriver.change`. `None` is a NULL callback.
    fn change(&self, job: &Arc<Job>, opts: &BlockJobChangeOptions) -> Option<Result<()>> {
        let _ = (job, opts);
        None
    }

    /// `BlockJobDriver.query`.
    fn query(&self, job: &Arc<Job>, info: &mut BlockJobInfo) {
        let _ = (job, info);
    }

    /// `.free`, before the job leaves the job list.
    fn free(&self, job: &Arc<Job>) {
        let _ = job;
    }

    /// The driver as `Any`, for tests.
    #[cfg_attr(not(test), allow(dead_code, reason = "only the tests use it so far"))]
    fn as_any(&self) -> Option<&dyn Any> {
        None
    }
}

/// `JobTxn`: jobs that complete or fail together.
#[derive(Default)]
pub(crate) struct JobTxn {
    st: Mutex<TxnState>,
}

#[derive(Default)]
struct TxnState {
    aborting: bool,
    /// Newest first.
    jobs: Vec<Arc<Job>>,
}

impl JobTxn {
    /// `job_txn_new()`.
    #[cfg_attr(not(test), allow(dead_code, reason = "only the tests use it so far"))]
    pub(crate) fn new() -> Arc<JobTxn> {
        Arc::new(JobTxn::default())
    }

    fn jobs(&self) -> Vec<Arc<Job>> {
        self.st.lock().unwrap().jobs.clone()
    }
}

/// The mutable part of a `Job`, protected by the job lock.
pub(crate) struct JobState {
    pub status: JobStatus,
    /// `job->ret`: 0 or a negative errno.
    pub ret: i32,
    pub err: Option<Error>,
    pub busy: bool,
    pub paused: bool,
    pub pause_count: u32,
    pub user_paused: bool,
    pub cancelled: bool,
    pub force_cancel: bool,
    pub deferred_to_main_loop: bool,
    /// `job->co != NULL`.
    pub started: bool,
    pub auto_finalize: bool,
    pub auto_dismiss: bool,
    txn: Option<Arc<JobTxn>>,
    /// When the sleep timer fires, if it is pending.
    deadline: Option<Instant>,
    /// Set when the job is entered while it waits.
    woken: bool,
    pub progress: Progress,
    refcnt: u32,
    /// `BlockJob.speed`.
    pub speed: i64,
    /// `BlockJob.iostatus`.
    pub iostatus: ruvm_qapi::types::BlockDeviceIoStatus,
}

impl JobState {
    /// `timer_pending(&job->sleep_timer)`.
    pub(crate) fn timer_pending(&self) -> bool {
        self.deadline.is_some()
    }
}

/// A `Job`.
pub(crate) struct Job {
    id: Option<String>,
    job_type: JobType,
    driver: OnceLock<Box<dyn JobDriver>>,
    st: Mutex<JobState>,
    cb: Mutex<Option<JobCb>>,
    /// The `BlockJob` part, for block jobs.
    pub(crate) bj: Option<BlockJob>,
    me: Weak<Job>,
}

impl std::fmt::Debug for Job {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Job").field("id", &self.id).field("type", &self.job_type).finish()
    }
}

/// The job list, `jobs`, newest first. Its mutex is `job_mutex`.
static JOBS: Mutex<Vec<Arc<Job>>> = Mutex::new(Vec::new());
/// Wakes jobs that wait to be entered.
static JOB_COND: Condvar = Condvar::new();

/// `job_lock()` until dropped.
pub(crate) struct JobLock {
    g: Option<MutexGuard<'static, Vec<Arc<Job>>>>,
}

/// `job_lock()`, `JOB_LOCK_GUARD()`.
pub(crate) fn job_lock() -> JobLock {
    let g = JOBS.lock().unwrap_or_else(|e| e.into_inner());
    note_job_lock(1);
    JobLock { g: Some(g) }
}

impl JobLock {
    /// `job_unlock(); f(); job_lock();`.
    pub(crate) fn unlocked<R>(&mut self, f: impl FnOnce() -> R) -> R {
        self.g = None;
        note_job_lock(-1);
        let r = f();
        self.g = Some(JOBS.lock().unwrap_or_else(|e| e.into_inner()));
        note_job_lock(1);
        r
    }

    fn wait(&mut self, timeout: Option<Duration>) {
        let g = self.g.take().expect("job lock held");
        let g = match timeout {
            Some(t) => JOB_COND.wait_timeout(g, t).unwrap_or_else(|e| e.into_inner()).0,
            None => JOB_COND.wait(g).unwrap_or_else(|e| e.into_inner()),
        };
        self.g = Some(g);
    }

    fn list(&mut self) -> &mut Vec<Arc<Job>> {
        self.g.as_mut().expect("job lock held")
    }

    /// `job_next_locked()` for every job, newest first.
    pub(crate) fn jobs(&mut self) -> Vec<Arc<Job>> {
        self.list().clone()
    }
}

impl Drop for JobLock {
    fn drop(&mut self) {
        if self.g.take().is_some() {
            note_job_lock(-1);
        }
    }
}

/// `job_get_locked()`.
pub(crate) fn job_get_locked(jl: &mut JobLock, id: &str) -> Option<Arc<Job>> {
    jl.list().iter().find(|j| j.id.as_deref() == Some(id)).cloned()
}

/// `id_wellformed()`.
fn id_wellformed(id: &str) -> bool {
    crate::graph::id_wellformed(id)
}

fn changed() {
    JOB_COND.notify_all();
    aio_wait_kick();
}

impl Job {
    /// `job_create()`: a job in CREATED state, paused once, in `txn` or a transaction of its
    /// own. `bj` is the block job part for block jobs.
    pub(crate) fn create(
        job_id: Option<&str>,
        job_type: JobType,
        txn: Option<Arc<JobTxn>>,
        flags: u32,
        cb: Option<JobCb>,
        bj: Option<BlockJob>,
    ) -> Result<Arc<Job>> {
        let mut jl = job_lock();
        match job_id {
            Some(id) => {
                if flags & JOB_INTERNAL != 0 {
                    return Err(Error::generic("Cannot specify job ID for internal job"));
                }
                if !id_wellformed(id) {
                    return Err(Error::generic(format!("Invalid job ID '{id}'")));
                }
                if job_get_locked(&mut jl, id).is_some() {
                    return Err(Error::generic(format!("Job ID '{id}' already in use")));
                }
            }
            None => {
                if flags & JOB_INTERNAL == 0 {
                    return Err(Error::generic("An explicit job ID is required"));
                }
            }
        }
        let job = Arc::new_cyclic(|me| Job {
            id: job_id.map(str::to_string),
            job_type,
            driver: OnceLock::new(),
            st: Mutex::new(JobState {
                status: JobStatus::Undefined,
                ret: 0,
                err: None,
                busy: false,
                paused: true,
                pause_count: 1,
                user_paused: false,
                cancelled: false,
                force_cancel: false,
                deferred_to_main_loop: false,
                started: false,
                auto_finalize: flags & JOB_MANUAL_FINALIZE == 0,
                auto_dismiss: flags & JOB_MANUAL_DISMISS == 0,
                txn: None,
                deadline: None,
                woken: false,
                progress: Progress::default(),
                refcnt: 1,
                speed: 0,
                iostatus: Default::default(),
            }),
            cb: Mutex::new(cb),
            bj,
            me: me.clone(),
        });
        job.state_transition_locked(&mut jl, JobStatus::Created);
        jl.list().insert(0, job.clone());
        // Single jobs are modeled as single-job transactions for sake of consolidating the
        // job management logic.
        let txn = txn.unwrap_or_default();
        txn_add_job_locked(&txn, &job);
        Ok(job)
    }

    /// The completion callback, for tests that set it once the job exists.
    #[cfg(test)]
    pub(crate) fn cb_slot(&self) -> MutexGuard<'_, Option<JobCb>> {
        self.cb.lock().unwrap()
    }

    /// Installs the driver. A job gets its driver once the code that starts it has set
    /// everything up, see the module documentation.
    pub(crate) fn set_driver(&self, driver: Box<dyn JobDriver>) {
        assert!(self.driver.set(driver).is_ok(), "job driver set twice");
    }

    /// The driver. Only valid once [`Job::set_driver`] ran.
    pub(crate) fn driver(&self) -> &dyn JobDriver {
        self.driver.get().expect("job has a driver").as_ref()
    }

    fn arc(&self) -> Arc<Job> {
        self.me.upgrade().expect("job is alive while borrowed")
    }

    /// The `Arc` this job lives in.
    pub(crate) fn self_arc(&self) -> Arc<Job> {
        self.arc()
    }

    /// The driver, if it is installed yet.
    pub(crate) fn driver_opt(&self) -> Option<&dyn JobDriver> {
        self.driver.get().map(|d| d.as_ref())
    }

    /// `job->id`, `None` for internal jobs.
    #[cfg_attr(not(test), allow(dead_code, reason = "only the tests use it so far"))]
    pub(crate) fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    /// `job->id` as `%s` prints it.
    pub(crate) fn id_str(&self) -> &str {
        self.id.as_deref().unwrap_or("(null)")
    }

    /// `job_type()`.
    pub(crate) fn job_type(&self) -> JobType {
        self.job_type
    }

    /// `job_is_internal()`.
    pub(crate) fn is_internal(&self) -> bool {
        self.id.is_none()
    }

    /// The state, for code that holds the job lock.
    pub(crate) fn s(&self) -> MutexGuard<'_, JobState> {
        self.st.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Runs `f` on the state with the job lock held.
    pub(crate) fn with_state<R>(&self, f: impl FnOnce(&mut JobState) -> R) -> R {
        let _jl = job_lock();
        f(&mut self.s())
    }

    /// `job->status`.
    #[cfg_attr(not(test), allow(dead_code, reason = "only the tests use it so far"))]
    pub(crate) fn status(&self) -> JobStatus {
        self.with_state(|s| s.status)
    }

    fn state_transition_locked(&self, _jl: &mut JobLock, s1: JobStatus) {
        let s0 = {
            let mut s = self.s();
            let s0 = s.status;
            assert!(
                JOB_STT[s0 as usize][s1 as usize],
                "job {} cannot go from {} to {}",
                self.id_str(),
                s0.as_str(),
                s1.as_str()
            );
            s.status = s1;
            s0
        };
        if !self.is_internal() && s1 != s0 {
            super::send_event(BlockEvent::JobStatusChange {
                id: self.id_str().to_string(),
                status: s1,
            });
        }
        changed();
    }

    /// `job_apply_verb_locked()`.
    pub(crate) fn apply_verb_locked(&self, _jl: &mut JobLock, verb: JobVerb) -> Result<()> {
        let s0 = self.s().status;
        if JOB_VERB_TABLE[verb as usize][s0 as usize] {
            return Ok(());
        }
        Err(Error::generic(format!(
            "Job '{}' in state '{}' cannot accept command verb '{}'",
            self.id_str(),
            s0.as_str(),
            verb.as_str()
        )))
    }

    /// `job_is_cancelled_locked()`: whether the job was cancelled for good.
    pub(crate) fn is_cancelled_locked(&self, _jl: &mut JobLock) -> bool {
        let s = self.s();
        // force_cancel may be true only if cancelled is true, too.
        assert!(s.cancelled || !s.force_cancel);
        s.force_cancel
    }

    /// `job_is_cancelled()`.
    pub(crate) fn is_cancelled(&self) -> bool {
        let mut jl = job_lock();
        self.is_cancelled_locked(&mut jl)
    }

    /// `job_cancel_requested()`.
    pub(crate) fn cancel_requested(&self) -> bool {
        self.with_state(|s| s.cancelled)
    }

    /// `job_is_paused()`.
    #[allow(dead_code, reason = "for tests and callers to come")]
    pub(crate) fn is_paused(&self) -> bool {
        self.with_state(|s| s.paused)
    }

    /// `job_is_ready_locked()`.
    pub(crate) fn is_ready_state(s: &JobState) -> bool {
        matches!(s.status, JobStatus::Ready | JobStatus::Standby)
    }

    /// `job_is_ready()`.
    pub(crate) fn is_ready(&self) -> bool {
        self.with_state(|s| Self::is_ready_state(s))
    }

    /// `job_is_completed_locked()`.
    pub(crate) fn is_completed_state(s: &JobState) -> bool {
        matches!(
            s.status,
            JobStatus::Waiting
                | JobStatus::Pending
                | JobStatus::Aborting
                | JobStatus::Concluded
                | JobStatus::Null
        )
    }

    /// `job_is_completed()`.
    pub(crate) fn is_completed(&self) -> bool {
        self.with_state(|s| Self::is_completed_state(s))
    }

    /// `job_ref_locked()`.
    pub(crate) fn ref_locked(&self, _jl: &mut JobLock) {
        self.s().refcnt += 1;
    }

    /// `job_unref_locked()`: frees the job with the last reference.
    pub(crate) fn unref_locked(&self, jl: &mut JobLock) {
        let last = {
            let mut s = self.s();
            s.refcnt -= 1;
            if s.refcnt == 0 {
                assert_eq!(s.status, JobStatus::Null);
                assert!(s.txn.is_none());
                s.deadline = None;
                true
            } else {
                false
            }
        };
        if !last {
            return;
        }
        let me = self.arc();
        jl.unlocked(|| {
            if let Some(d) = self.driver.get() {
                d.free(&me);
            }
            if let Some(bj) = &self.bj {
                bj.free(&me);
            }
        });
        jl.list().retain(|j| !std::ptr::eq(j.as_ref(), self));
    }

    /// `job_progress_update()`.
    pub(crate) fn progress_update(&self, done: u64) {
        self.with_state(|s| s.progress.work_done(done));
    }

    /// `job_progress_set_remaining()`.
    pub(crate) fn progress_set_remaining(&self, remaining: u64) {
        self.with_state(|s| s.progress.set_remaining(remaining));
    }

    /// `job_progress_increase_remaining()`.
    pub(crate) fn progress_increase_remaining(&self, delta: u64) {
        self.with_state(|s| s.progress.increase_remaining(delta));
    }

    /// `job_enter_cond_locked()`: wakes the job if it is waiting and `cond` allows it.
    pub(crate) fn enter_cond_locked(&self, _jl: &mut JobLock, cond: Option<fn(&JobState) -> bool>) {
        let mut s = self.s();
        if !s.started || s.deferred_to_main_loop || s.busy {
            return;
        }
        if let Some(f) = cond {
            if !f(&s) {
                return;
            }
        }
        s.deadline = None;
        s.busy = true;
        s.woken = true;
        drop(s);
        JOB_COND.notify_all();
    }

    /// `job_enter()`.
    pub(crate) fn enter(&self) {
        let mut jl = job_lock();
        self.enter_cond_locked(&mut jl, None);
    }

    /// `job_do_yield_locked()`: waits until the job is entered again, or until `deadline`.
    fn do_yield_locked(&self, jl: &mut JobLock, deadline: Option<Instant>) {
        {
            let mut s = self.s();
            s.deadline = deadline;
            s.busy = false;
            s.woken = false;
        }
        // job_event_idle_locked(): block_job_on_idle_locked() kicks the waiters.
        changed();
        loop {
            let (woken, deadline) = {
                let s = self.s();
                (s.woken, s.deadline)
            };
            if woken {
                break;
            }
            match deadline {
                Some(d) => {
                    let now = Instant::now();
                    if now >= d {
                        // job_sleep_timer_cb()
                        self.enter_cond_locked(jl, None);
                        continue;
                    }
                    jl.wait(Some(d - now));
                }
                None => jl.wait(None),
            }
        }
        assert!(self.s().busy);
    }

    fn should_pause(&self) -> bool {
        self.s().pause_count > 0
    }

    /// `job_pause_point_locked()`.
    fn pause_point_locked(&self, jl: &mut JobLock) {
        assert!(self.s().started);
        if !self.should_pause() {
            return;
        }
        if self.is_cancelled_locked(jl) {
            return;
        }
        let me = self.arc();
        jl.unlocked(|| self.driver().pause(&me));
        if self.should_pause() && !self.is_cancelled_locked(jl) {
            let status = self.s().status;
            let next =
                if status == JobStatus::Ready { JobStatus::Standby } else { JobStatus::Paused };
            self.state_transition_locked(jl, next);
            self.s().paused = true;
            self.do_yield_locked(jl, None);
            self.s().paused = false;
            self.state_transition_locked(jl, status);
        }
        jl.unlocked(|| self.driver().resume(&me));
    }

    /// `job_pause_point()`: pauses here if the job should pause.
    pub(crate) fn pause_point(&self) {
        let mut jl = job_lock();
        self.pause_point_locked(&mut jl);
    }

    /// `job_yield()`: waits until entered, unless cancelled.
    pub(crate) fn yield_(&self) {
        let mut jl = job_lock();
        assert!(self.s().busy);
        // Check cancellation *before* setting busy = false, too!
        if self.is_cancelled_locked(&mut jl) {
            return;
        }
        if !self.should_pause() {
            self.do_yield_locked(&mut jl, None);
        }
        self.pause_point_locked(&mut jl);
    }

    /// [`Job::yield_`], unless `done` holds. `done` is checked under the job lock, so a
    /// callback that makes it true and then calls [`Job::enter`] cannot be missed, as it
    /// can be between a check before `yield_()` and the yield.
    pub(crate) fn yield_unless(&self, done: impl Fn() -> bool) {
        let mut jl = job_lock();
        assert!(self.s().busy);
        if self.is_cancelled_locked(&mut jl) || done() {
            return;
        }
        if !self.should_pause() {
            self.do_yield_locked(&mut jl, None);
        }
        self.pause_point_locked(&mut jl);
    }

    /// `job_sleep_ns()`: waits `ns` nanoseconds or until entered, unless cancelled.
    pub(crate) fn sleep_ns(&self, ns: i64) {
        let mut jl = job_lock();
        assert!(self.s().busy);
        // Check cancellation *before* setting busy = false, too!
        if self.is_cancelled_locked(&mut jl) {
            return;
        }
        if !self.should_pause() {
            let d = Instant::now() + Duration::from_nanos(ns.max(0) as u64);
            self.do_yield_locked(&mut jl, Some(d));
        }
        self.pause_point_locked(&mut jl);
    }

    /// `job_pause_locked()`.
    pub(crate) fn pause_locked(&self, jl: &mut JobLock) {
        let paused = {
            let mut s = self.s();
            s.pause_count += 1;
            s.paused
        };
        if !paused {
            self.enter_cond_locked(jl, None);
        }
    }

    /// `job_pause()`.
    pub(crate) fn pause(&self) {
        let mut jl = job_lock();
        self.pause_locked(&mut jl);
    }

    /// `job_resume_locked()`.
    pub(crate) fn resume_locked(&self, jl: &mut JobLock) {
        {
            let mut s = self.s();
            assert!(s.pause_count > 0);
            s.pause_count -= 1;
            if s.pause_count > 0 {
                return;
            }
        }
        // kick only if no timer is pending
        self.enter_cond_locked(jl, Some(|s| s.deadline.is_none()));
    }

    /// `job_resume()`.
    pub(crate) fn resume(&self) {
        let mut jl = job_lock();
        self.resume_locked(&mut jl);
    }

    /// `job_user_pause_locked()`.
    pub(crate) fn user_pause_locked(&self, jl: &mut JobLock) -> Result<()> {
        self.apply_verb_locked(jl, JobVerb::Pause)?;
        if self.s().user_paused {
            return Err(Error::generic("Job is already paused"));
        }
        self.s().user_paused = true;
        self.pause_locked(jl);
        Ok(())
    }

    /// `job_user_paused_locked()`.
    pub(crate) fn user_paused_locked(&self, _jl: &mut JobLock) -> bool {
        self.s().user_paused
    }

    /// The `.user_resume` callbacks: `block_job_user_resume()` for block jobs, then the
    /// driver's own.
    fn driver_user_resume(&self, jl: &mut JobLock) {
        if self.bj.is_some() {
            self.iostatus_reset_locked(jl);
        }
        let me = self.arc();
        if let Some(d) = self.driver.get() {
            jl.unlocked(|| d.user_resume(&me));
        }
    }

    /// `block_job_iostatus_reset_locked()`.
    pub(crate) fn iostatus_reset_locked(&self, _jl: &mut JobLock) {
        let mut s = self.s();
        if s.iostatus == ruvm_qapi::types::BlockDeviceIoStatus::Ok {
            return;
        }
        assert!(s.user_paused && s.pause_count > 0);
        s.iostatus = ruvm_qapi::types::BlockDeviceIoStatus::Ok;
    }

    /// `job_user_resume_locked()`.
    pub(crate) fn user_resume_locked(&self, jl: &mut JobLock) -> Result<()> {
        {
            let s = self.s();
            if !s.user_paused || s.pause_count == 0 {
                return Err(Error::generic("Can't resume a job that was not paused"));
            }
        }
        self.apply_verb_locked(jl, JobVerb::Resume)?;
        self.driver_user_resume(jl);
        self.s().user_paused = false;
        self.resume_locked(jl);
        Ok(())
    }

    /// `job_do_dismiss_locked()`.
    fn do_dismiss_locked(&self, jl: &mut JobLock) {
        {
            let mut s = self.s();
            s.busy = false;
            s.paused = false;
            s.deferred_to_main_loop = true;
        }
        txn_del_job_locked(self);
        self.state_transition_locked(jl, JobStatus::Null);
        self.unref_locked(jl);
    }

    /// `job_dismiss_locked()`.
    pub(crate) fn dismiss_locked(&self, jl: &mut JobLock) -> Result<()> {
        // similarly to _complete, this is QMP-interface only.
        assert!(self.id.is_some());
        self.apply_verb_locked(jl, JobVerb::Dismiss)?;
        self.do_dismiss_locked(jl);
        Ok(())
    }

    /// `job_early_fail()`: drops a job that never started.
    pub(crate) fn early_fail(&self) {
        let mut jl = job_lock();
        assert_eq!(self.s().status, JobStatus::Created);
        self.do_dismiss_locked(&mut jl);
    }

    /// `job_conclude_locked()`.
    fn conclude_locked(&self, jl: &mut JobLock) {
        self.state_transition_locked(jl, JobStatus::Concluded);
        let (auto_dismiss, started) = {
            let s = self.s();
            (s.auto_dismiss, s.started)
        };
        if auto_dismiss || !started {
            self.do_dismiss_locked(jl);
        }
    }

    /// `job_update_rc_locked()`.
    fn update_rc_locked(&self, jl: &mut JobLock) {
        if self.s().ret == 0 && self.is_cancelled_locked(jl) {
            self.s().ret = -ECANCELED;
        }
        let failed = {
            let mut s = self.s();
            if s.ret != 0 && s.err.is_none() {
                s.err = Some(Error::generic(strerror(-s.ret)));
            }
            s.ret != 0
        };
        if failed {
            self.state_transition_locked(jl, JobStatus::Aborting);
        }
    }

    /// `job_finalize_single_locked()`.
    fn finalize_single_locked(&self, jl: &mut JobLock) -> i32 {
        assert!(Self::is_completed_state(&self.s()));
        // Ensure abort is called for late-transactional failures
        self.update_rc_locked(jl);
        let ret = self.s().ret;
        let me = self.arc();
        let cb = self.cb.lock().unwrap().take();
        jl.unlocked(|| {
            if let Some(d) = self.driver.get() {
                if ret == 0 {
                    d.commit(&me);
                } else {
                    d.abort(&me);
                }
                d.clean(&me);
            }
            if let Some(cb) = cb {
                cb(ret);
            }
        });
        // Emit events only if we actually started
        if self.s().started {
            if self.is_cancelled_locked(jl) {
                self.event_cancelled_locked(jl);
            } else {
                self.event_completed_locked(jl);
            }
        }
        txn_del_job_locked(self);
        self.conclude_locked(jl);
        0
    }

    /// `job_cancel_async_locked()`.
    fn cancel_async_locked(&self, jl: &mut JobLock, force: bool) {
        let me = self.arc();
        let force = match self.driver.get() {
            Some(d) => jl.unlocked(|| d.cancel(&me, force)).unwrap_or(true),
            // No .cancel() means the job will behave as if force-cancelled
            None => true,
        };
        if self.s().user_paused {
            // Do not call job_enter here, the caller will handle it.
            self.driver_user_resume(jl);
            let mut s = self.s();
            s.user_paused = false;
            assert!(s.pause_count > 0);
            s.pause_count -= 1;
        }
        // Ignore soft cancel requests after the job is already done (We will still invoke
        // job->driver->cancel() above, but if the job driver supports soft cancelling and the
        // job is done, that should be a no-op, too. We still call it so it can override
        // @force.)
        let mut s = self.s();
        if force || !s.deferred_to_main_loop {
            s.cancelled = true;
            // To prevent 'force == false' overriding a previous 'force == true'
            s.force_cancel |= force;
        }
    }

    fn txn(&self) -> Arc<JobTxn> {
        self.s().txn.clone().expect("job is in a transaction")
    }

    /// `job_completed_txn_abort_locked()`.
    fn completed_txn_abort_locked(&self, jl: &mut JobLock) {
        let txn = self.txn();
        {
            let mut t = txn.st.lock().unwrap();
            if t.aborting {
                // We are cancelled by another job, which will handle everything.
                return;
            }
            t.aborting = true;
        }
        self.ref_locked(jl);
        // Other jobs are effectively cancelled by us, set the status for them; this job,
        // however, may or may not be cancelled, depending on the caller, so leave it.
        for other in txn.jobs() {
            if !std::ptr::eq(other.as_ref(), self) {
                // This is a transaction: If one job failed, no result will matter.
                // Therefore, pass force=true to terminate all other jobs as quickly as
                // possible.
                other.cancel_async_locked(jl, true);
            }
        }
        loop {
            let Some(other) = txn.jobs().first().cloned() else {
                break;
            };
            if !Self::is_completed_state(&other.s()) {
                assert!(other.s().cancelled);
                let _ = other.finish_sync_locked(jl, None);
            }
            other.finalize_single_locked(jl);
        }
        self.unref_locked(jl);
    }

    /// `job_prepare_locked()`.
    fn prepare_locked(&self, jl: &mut JobLock) -> i32 {
        if self.s().ret == 0 {
            let me = self.arc();
            let r = jl.unlocked(|| self.driver.get().and_then(|d| d.prepare(&me)));
            if let Some(r) = r {
                {
                    let mut s = self.s();
                    match r {
                        Ok(()) => s.ret = 0,
                        Err(e) => {
                            s.ret = -e.errno;
                            if s.err.is_none() {
                                s.err = e.err;
                            }
                        }
                    }
                }
                self.update_rc_locked(jl);
            }
        }
        self.s().ret
    }

    /// `job_do_finalize_locked()`.
    fn do_finalize_locked(&self, jl: &mut JobLock) {
        // prepare the transaction to complete
        let rc = txn_apply_locked(jl, self, |jl, j| j.prepare_locked(jl));
        if rc != 0 {
            self.completed_txn_abort_locked(jl);
        } else {
            txn_apply_locked(jl, self, |jl, j| j.finalize_single_locked(jl));
        }
    }

    /// `job_finalize_locked()`.
    pub(crate) fn finalize_locked(&self, jl: &mut JobLock) -> Result<()> {
        assert!(self.id.is_some());
        self.apply_verb_locked(jl, JobVerb::Finalize)?;
        self.do_finalize_locked(jl);
        Ok(())
    }

    /// `job_transition_to_pending_locked()`.
    fn transition_to_pending_locked(&self, jl: &mut JobLock) -> i32 {
        self.state_transition_locked(jl, JobStatus::Pending);
        if !self.s().auto_finalize {
            self.event_pending_locked(jl);
        }
        0
    }

    /// `job_transition_to_ready()`.
    pub(crate) fn transition_to_ready(&self) {
        let mut jl = job_lock();
        self.state_transition_locked(&mut jl, JobStatus::Ready);
        self.event_ready_locked(&mut jl);
    }

    /// `job_completed_txn_success_locked()`.
    fn completed_txn_success_locked(&self, jl: &mut JobLock) {
        let txn = self.txn();
        self.state_transition_locked(jl, JobStatus::Waiting);
        // Successful completion, see if there are other running jobs in this txn.
        for other in txn.jobs() {
            let s = other.s();
            if !Self::is_completed_state(&s) {
                return;
            }
            assert_eq!(s.ret, 0);
        }
        txn_apply_locked(jl, self, |jl, j| j.transition_to_pending_locked(jl));
        // If no jobs need manual finalization, automatically do so
        if txn_apply_locked(jl, self, |_, j| i32::from(!j.s().auto_finalize)) == 0 {
            self.do_finalize_locked(jl);
        }
    }

    /// `job_completed_locked()`.
    fn completed_locked(&self, jl: &mut JobLock) {
        assert!(!Self::is_completed_state(&self.s()));
        self.update_rc_locked(jl);
        if self.s().ret != 0 {
            self.completed_txn_abort_locked(jl);
        } else {
            self.completed_txn_success_locked(jl);
        }
    }

    /// `job_exit()`, the bottom half that runs once `run` returned.
    fn exit(self: &Arc<Self>) {
        let mut jl = job_lock();
        self.ref_locked(&mut jl);
        // This is a lie, we're not quiescent, but still doing the completion callbacks.
        // However, completion callbacks tend to involve operations that drain block nodes,
        // and if .drained_poll still returned true, we would deadlock.
        self.s().busy = false;
        changed();
        self.completed_locked(&mut jl);
        self.unref_locked(&mut jl);
    }

    /// `job_co_entry()`, on the job's thread.
    fn co_entry(self: Arc<Self>) {
        {
            let mut jl = job_lock();
            self.pause_point_locked(&mut jl);
        }
        let r = self.driver().run(&self);
        {
            let _jl = job_lock();
            let mut s = self.s();
            match r {
                Ok(()) => s.ret = 0,
                Err(e) => {
                    s.ret = -e.errno;
                    if e.err.is_some() {
                        s.err = e.err;
                    }
                }
            }
            s.deferred_to_main_loop = true;
            s.busy = true;
        }
        let me = self.clone();
        schedule_bh(move || me.exit());
    }

    /// `job_start()`: runs the job on a thread of its own.
    pub(crate) fn start(self: &Arc<Self>) {
        {
            let mut jl = job_lock();
            {
                let mut s = self.s();
                assert!(!s.started && s.paused && self.driver.get().is_some());
                s.started = true;
                s.pause_count -= 1;
                s.busy = true;
                s.paused = false;
            }
            self.state_transition_locked(&mut jl, JobStatus::Running);
        }
        let me = self.clone();
        let name = format!("job {}", self.id_str());
        std::thread::Builder::new()
            .name(name)
            .spawn(move || me.co_entry())
            .expect("cannot spawn a job thread");
    }

    /// `job_cancel_locked()`.
    pub(crate) fn cancel_locked(&self, jl: &mut JobLock, force: bool) {
        if self.s().status == JobStatus::Concluded {
            self.do_dismiss_locked(jl);
            return;
        }
        self.cancel_async_locked(jl, force);
        let (started, deferred) = {
            let s = self.s();
            (s.started, s.deferred_to_main_loop)
        };
        if !started {
            self.completed_locked(jl);
        } else if deferred {
            // job_cancel_async() ignores soft-cancel requests for jobs that are already done
            // (i.e. deferred to the main loop). We have to check again whether the job is
            // really cancelled.
            if self.is_cancelled_locked(jl) {
                self.completed_txn_abort_locked(jl);
            }
        } else {
            self.enter_cond_locked(jl, None);
        }
    }

    /// `job_user_cancel_locked()`.
    pub(crate) fn user_cancel_locked(&self, jl: &mut JobLock, force: bool) -> Result<()> {
        self.apply_verb_locked(jl, JobVerb::Cancel)?;
        self.cancel_locked(jl, force);
        Ok(())
    }

    /// `job_cancel_sync_locked()`.
    pub(crate) fn cancel_sync_locked(&self, jl: &mut JobLock, force: bool) -> i32 {
        let f: FinishFn = if force {
            |jl, j| {
                j.cancel_locked(jl, true);
                Ok(())
            }
        } else {
            |jl, j| {
                j.cancel_locked(jl, false);
                Ok(())
            }
        };
        self.finish_sync_locked(jl, Some(f)).unwrap_or_else(|_| unreachable!())
    }

    /// `job_cancel_sync()`. Takes the BQL.
    #[cfg_attr(not(test), allow(dead_code, reason = "only the tests use it so far"))]
    pub(crate) fn cancel_sync(&self, force: bool) -> i32 {
        let _bql = super::main_loop::bql_lock();
        let mut jl = job_lock();
        self.cancel_sync_locked(&mut jl, force)
    }

    /// `job_complete_locked()`.
    pub(crate) fn complete_locked(&self, jl: &mut JobLock) -> Result<()> {
        // Should not be reachable via external interface for internal jobs
        assert!(self.id.is_some());
        self.apply_verb_locked(jl, JobVerb::Complete)?;
        let has = self.driver.get().is_some_and(|d| d.has_complete());
        if self.s().cancelled || !has {
            return Err(Error::generic(format!(
                "The active block job '{}' cannot be completed",
                self.id_str()
            )));
        }
        let me = self.arc();
        jl.unlocked(|| self.driver().complete(&me))
    }

    /// `job_complete_sync_locked()`.
    #[cfg_attr(not(test), allow(dead_code, reason = "only the tests use it so far"))]
    pub(crate) fn complete_sync_locked(&self, jl: &mut JobLock) -> Result<i32> {
        self.finish_sync_locked(jl, Some(|jl, j| j.complete_locked(jl)))
    }

    /// `job_finish_sync_locked()`: runs `finish` and waits until the job completed. Returns
    /// the job's return value, `-ECANCELED` for a cancelled job that returned 0.
    pub(crate) fn finish_sync_locked(
        &self,
        jl: &mut JobLock,
        finish: Option<FinishFn>,
    ) -> Result<i32> {
        self.ref_locked(jl);
        if let Some(f) = finish {
            if let Err(e) = f(jl, self) {
                self.unref_locked(jl);
                return Err(e);
            }
        }
        jl.unlocked(|| {
            aio_wait_while(|| {
                self.enter();
                !self.is_completed()
            })
        });
        let ret = {
            let cancelled = self.is_cancelled_locked(jl);
            let s = self.s();
            if cancelled && s.ret == 0 { -ECANCELED } else { s.ret }
        };
        self.unref_locked(jl);
        Ok(ret)
    }

    /// `job_query_single_locked()`.
    pub(crate) fn query_locked(&self, _jl: &mut JobLock) -> JobInfo {
        assert!(!self.is_internal());
        let s = self.s();
        JobInfo {
            id: self.id_str().to_string(),
            type_: self.job_type,
            status: s.status,
            current_progress: s.progress.current as i64,
            total_progress: s.progress.total as i64,
            error: s.err.as_ref().map(|e| e.message().to_string()),
        }
    }
}

/// What `job_finish_sync_locked()` runs first.
pub(crate) type FinishFn = fn(&mut JobLock, &Job) -> Result<()>;

/// `job_txn_add_job_locked()`.
pub(crate) fn txn_add_job_locked(txn: &Arc<JobTxn>, job: &Arc<Job>) {
    let mut s = job.s();
    assert!(s.txn.is_none());
    s.txn = Some(txn.clone());
    txn.st.lock().unwrap().jobs.insert(0, job.clone());
}

/// `job_txn_del_job_locked()`.
fn txn_del_job_locked(job: &Job) {
    let txn = job.s().txn.take();
    if let Some(t) = txn {
        t.st.lock().unwrap().jobs.retain(|j| !std::ptr::eq(j.as_ref(), job));
    }
}

/// `job_txn_apply_locked()`: runs `f` on every job of the transaction of `job` until one
/// returns non-zero.
fn txn_apply_locked(jl: &mut JobLock, job: &Job, f: impl Fn(&mut JobLock, &Job) -> i32) -> i32 {
    let txn = job.txn();
    job.ref_locked(jl);
    let mut rc = 0;
    for other in txn.jobs() {
        rc = f(jl, &other);
        if rc != 0 {
            break;
        }
    }
    job.unref_locked(jl);
    rc
}

/// `job_cancel_sync_all()`.
#[allow(dead_code, reason = "for the system emulator's shutdown")]
pub(crate) fn cancel_sync_all() {
    let _bql = super::main_loop::bql_lock();
    let mut jl = job_lock();
    while let Some(job) = jl.list().first().cloned() {
        job.cancel_sync_locked(&mut jl, true);
    }
}
