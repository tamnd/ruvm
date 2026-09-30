// SPDX-License-Identifier: GPL-2.0-or-later

//! tests/unit/test-blockjob.c.

use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ruvm_base::Result;
use ruvm_qapi::types::{JobStatus, JobType};

use super::*;
use crate::job::block_job::{BlockJobParams, block_job_create};
use crate::job::core::{JOB_DEFAULT, JOB_MANUAL_DISMISS, JOB_MANUAL_FINALIZE, JobDriver, JobErr};
use crate::job::main_loop::bql_lock;

/// `test_block_job_driver`: nothing to run.
struct Nop;

impl JobDriver for Nop {
    fn run(&self, _job: &Arc<Job>) -> Result<(), JobErr> {
        Ok(())
    }
}

/// `mk_job()`.
fn mk_job(
    g: &BlockGraph,
    blk: &BlockBackend,
    id: Option<&str>,
    should_succeed: bool,
    flags: u32,
) -> Option<Arc<Job>> {
    let r = block_job_create(
        g,
        &blk.root().unwrap(),
        BlockJobParams {
            job_id: id,
            job_type: JobType::Commit,
            txn: None,
            perm: 0,
            shared: BLK_PERM_ALL,
            speed: 0,
            flags,
            cb: Some(Box::new(|_| {})),
        },
    );
    if should_succeed {
        let job = r.unwrap();
        job.set_driver(Box::new(Nop));
        assert_eq!(job.id().unwrap(), id.unwrap_or(blk.name()));
        Some(job)
    } else {
        assert!(r.is_err());
        None
    }
}

fn do_test_id(g: &BlockGraph, blk: &BlockBackend, id: Option<&str>, ok: bool) -> Option<Arc<Job>> {
    mk_job(g, blk, id, ok, JOB_DEFAULT)
}

#[test]
fn job_ids() {
    let _s = serial();
    let g = BlockGraph::new();
    let blk0 = create_blk(&g, "ids0", None);
    let blk1 = create_blk(&g, "ids1", Some("drive1"));
    let blk2 = create_blk(&g, "ids2", Some("drive2"));

    // No job ID provided and the block backend has no name
    do_test_id(&g, &blk0, None, false);

    // These are all invalid job IDs
    for id in ["0id", "", "   ", "123", "_id", "-id", ".id", "#id"] {
        do_test_id(&g, &blk0, Some(id), false);
    }

    // This one is valid
    let j0 = do_test_id(&g, &blk0, Some("id0"), true).unwrap();

    // We can have two jobs in the same BDS
    let j1 = do_test_id(&g, &blk0, Some("id1"), true).unwrap();
    j1.early_fail();

    // Duplicate job IDs are not allowed
    do_test_id(&g, &blk1, Some("id0"), false);

    // But once job[0] finishes we can reuse its ID
    j0.early_fail();
    let j1 = do_test_id(&g, &blk1, Some("id0"), true).unwrap();

    // No job ID specified, defaults to the backend name ('drive1')
    j1.early_fail();
    let j1 = do_test_id(&g, &blk1, None, true).unwrap();

    // Duplicate job ID
    do_test_id(&g, &blk2, Some("drive1"), false);

    // The ID of job[2] would default to 'drive2' but it is already in use
    let j0 = do_test_id(&g, &blk0, Some("drive2"), true).unwrap();
    do_test_id(&g, &blk2, None, false);

    // This one is valid
    let j2 = do_test_id(&g, &blk2, Some("id_2"), true).unwrap();

    j0.early_fail();
    j1.early_fail();
    j2.early_fail();
    assert!(g.query_jobs().is_empty());
    let root = blk0.root().unwrap();
    assert_eq!(root.parent_count(), 1, "the jobs let go of the node");
}

#[test]
fn job_create_errors() {
    use crate::job::core::JOB_INTERNAL;
    let _s = serial();
    let err = |id: Option<&str>, flags| {
        Job::create(id, JobType::Commit, None, flags, None, None).unwrap_err().message().to_string()
    };
    assert_eq!(err(Some("x"), JOB_INTERNAL), "Cannot specify job ID for internal job");
    assert_eq!(err(Some("0x"), 0), "Invalid job ID '0x'");
    assert_eq!(err(None, 0), "An explicit job ID is required");
}

/// `CancelJob`.
#[derive(Default)]
struct CancelJob {
    should_converge: AtomicBool,
    should_complete: AtomicBool,
}

impl JobDriver for CancelJob {
    fn run(&self, job: &Arc<Job>) -> Result<(), JobErr> {
        while !self.should_complete.load(Ordering::SeqCst) {
            if job.is_cancelled() {
                return Ok(());
            }
            if !job.is_ready() && self.should_converge.load(Ordering::SeqCst) {
                job.transition_to_ready();
            }
            job.sleep_ns(100_000);
        }
        Ok(())
    }

    fn has_complete(&self) -> bool {
        true
    }

    fn complete(&self, _job: &Arc<Job>) -> Result<()> {
        self.should_complete.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn as_any(&self) -> Option<&dyn Any> {
        Some(self)
    }
}

fn cj(job: &Job) -> &CancelJob {
    job.driver().as_any().unwrap().downcast_ref().unwrap()
}

struct Common {
    g: BlockGraph,
    _blk: Arc<BlockBackend>,
    job: Arc<Job>,
}

/// `create_common()`.
fn create_common(node: &str) -> Common {
    let g = BlockGraph::new();
    let blk = create_blk(&g, node, None);
    let job = block_job_create(
        &g,
        &blk.root().unwrap(),
        BlockJobParams {
            job_id: Some("Steve"),
            job_type: JobType::Commit,
            txn: None,
            perm: 0,
            shared: BLK_PERM_ALL,
            speed: 0,
            flags: JOB_MANUAL_FINALIZE | JOB_MANUAL_DISMISS,
            cb: None,
        },
    )
    .unwrap();
    job.set_driver(Box::<CancelJob>::default());
    let mut jl = job_lock();
    job.ref_locked(&mut jl);
    assert_eq!(job.s().status, JobStatus::Created);
    drop(jl);
    take_events("Steve");
    Common { g, _blk: blk, job }
}

/// `cancel_common()`.
fn cancel_common(c: Common) {
    let job = &c.job;
    let sts = job.status();
    job.cancel_sync(true);
    let mut jl = job_lock();
    if sts != JobStatus::Created && sts != JobStatus::Concluded {
        job.dismiss_locked(&mut jl).unwrap();
    }
    assert_eq!(job.s().status, JobStatus::Null);
    job.unref_locked(&mut jl);
    drop(jl);
    assert!(c.g.query_jobs().is_empty());
    take_events("Steve");
}

#[test]
fn cancel_created() {
    let _s = serial();
    let c = create_common("cc0");
    cancel_common(c);
}

#[test]
fn cancel_running() {
    let _s = serial();
    let c = create_common("cc1");
    c.job.start();
    assert_status(&c.job, JobStatus::Running);
    cancel_common(c);
}

#[test]
fn cancel_paused() {
    let _s = serial();
    let c = create_common("cc2");
    c.job.start();
    {
        let mut jl = job_lock();
        assert_eq!(c.job.s().status, JobStatus::Running);
        c.job.user_pause_locked(&mut jl).unwrap();
    }
    c.job.enter();
    wait_status(&c.job, JobStatus::Paused);
    cancel_common(c);
}

#[test]
fn cancel_ready() {
    let _s = serial();
    let c = create_common("cc3");
    c.job.start();
    assert_status(&c.job, JobStatus::Running);
    cj(&c.job).should_converge.store(true, Ordering::SeqCst);
    c.job.enter();
    wait_status(&c.job, JobStatus::Ready);
    cancel_common(c);
}

#[test]
fn cancel_standby() {
    let _s = serial();
    let c = create_common("cc4");
    c.job.start();
    cj(&c.job).should_converge.store(true, Ordering::SeqCst);
    c.job.enter();
    wait_status(&c.job, JobStatus::Ready);
    {
        let mut jl = job_lock();
        c.job.user_pause_locked(&mut jl).unwrap();
    }
    c.job.enter();
    wait_status(&c.job, JobStatus::Standby);
    cancel_common(c);
}

/// Up to PENDING, with the BQL held so that `job_exit()` waits for the "main loop".
fn to_pending(c: &Common) {
    c.job.start();
    cj(&c.job).should_converge.store(true, Ordering::SeqCst);
    c.job.enter();
    wait_status(&c.job, JobStatus::Ready);
    let _bql = bql_lock();
    {
        let mut jl = job_lock();
        c.job.complete_locked(&mut jl).unwrap();
    }
    c.job.enter();
    wait_for("deferred", || c.job.with_state(|s| s.deferred_to_main_loop));
    assert_status(&c.job, JobStatus::Ready);
    wait_status(&c.job, JobStatus::Pending);
}

#[test]
fn cancel_pending() {
    let _s = serial();
    let c = create_common("cc5");
    to_pending(&c);
    let evs = statuses(&take_events("Steve"));
    assert_eq!(evs, [JobStatus::Running, JobStatus::Ready, JobStatus::Waiting, JobStatus::Pending]);
    cancel_common(c);
}

#[test]
fn cancel_concluded() {
    let _s = serial();
    let c = create_common("cc6");
    to_pending(&c);
    {
        let _bql = bql_lock();
        let mut jl = job_lock();
        c.job.finalize_locked(&mut jl).unwrap();
        assert_eq!(c.job.s().status, JobStatus::Concluded);
    }
    let evs = take_events("Steve");
    assert!(evs.iter().any(|e| matches!(e, BlockEvent::BlockJobCompleted { error: None, .. })));
    cancel_common(c);
}
