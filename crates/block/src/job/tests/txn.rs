// SPDX-License-Identifier: GPL-2.0-or-later

//! tests/unit/test-blockjob-txn.c.

use std::sync::Arc;
use std::sync::atomic::{AtomicI32, AtomicU32, Ordering};

use ruvm_base::Result;
use ruvm_qapi::types::JobType;

use super::*;
use crate::job::block_job::{BlockJobParams, block_job_create};
use crate::job::core::{ECANCELED, JOB_DEFAULT, JobDriver, JobErr, JobTxn};

const EINPROGRESS: i32 = -libc::EINPROGRESS;
const EIO: i32 = -libc::EIO;

/// `TestBlockJob`.
struct TestBlockJob {
    iterations: AtomicU32,
    use_timer: bool,
    rc: i32,
}

impl JobDriver for TestBlockJob {
    fn run(&self, job: &Arc<Job>) -> Result<(), JobErr> {
        loop {
            let it = self.iterations.load(Ordering::SeqCst);
            self.iterations.store(it.wrapping_sub(1), Ordering::SeqCst);
            if it == 0 {
                break;
            }
            if self.use_timer {
                job.sleep_ns(0);
            } else {
                job.yield_();
            }
            if job.is_cancelled() {
                break;
            }
        }
        if self.rc == 0 { Ok(()) } else { Err(JobErr::errno(-self.rc)) }
    }
}

static COUNTER: AtomicU32 = AtomicU32::new(0);

/// `test_block_job_start()`: a job that completes with `rc` after `iterations` yields.
fn test_block_job_start(
    g: &BlockGraph,
    iterations: u32,
    use_timer: bool,
    rc: i32,
    result: &Arc<AtomicI32>,
    txn: &Arc<JobTxn>,
) -> Arc<Job> {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let node = format!("txn{n}");
    add(g, &format!(r#"{{"driver": "null-co", "node-name": "{node}", "read-zeroes": true}}"#));
    let bs = g.find_node(&node).unwrap();
    let job_id = format!("txnjob{n}");
    let r = result.clone();
    let job = block_job_create(
        g,
        &bs,
        BlockJobParams {
            job_id: Some(&job_id),
            job_type: JobType::Commit,
            txn: Some(txn.clone()),
            perm: 0,
            shared: BLK_PERM_ALL,
            speed: 0,
            flags: JOB_DEFAULT,
            cb: None,
        },
    )
    .unwrap();
    // test_block_job_cb()
    let weak = Arc::downgrade(&job);
    *job.cb_slot() = Some(Box::new(move |mut ret| {
        if ret == 0 && weak.upgrade().is_some_and(|j| j.is_cancelled()) {
            ret = -ECANCELED;
        }
        r.store(ret, Ordering::SeqCst);
    }));
    job.set_driver(Box::new(TestBlockJob {
        iterations: AtomicU32::new(iterations),
        use_timer,
        rc,
    }));
    job
}

fn wait_results(rs: &[&Arc<AtomicI32>]) {
    wait_for("the results", || rs.iter().all(|r| r.load(Ordering::SeqCst) != EINPROGRESS));
}

fn single_job(expected: i32) {
    let g = BlockGraph::new();
    let result = Arc::new(AtomicI32::new(EINPROGRESS));
    let txn = JobTxn::new();
    let job = test_block_job_start(&g, 1, true, expected, &result, &txn);
    job.start();
    {
        let mut jl = job_lock();
        if expected == -ECANCELED {
            job.cancel_locked(&mut jl, false);
        }
    }
    wait_results(&[&result]);
    assert_eq!(result.load(Ordering::SeqCst), expected);
}

#[test]
fn single_success() {
    let _s = serial();
    single_job(0);
}

#[test]
fn single_failure() {
    let _s = serial();
    single_job(EIO);
}

#[test]
fn single_cancel() {
    let _s = serial();
    single_job(-ECANCELED);
}

fn pair_jobs(mut expected1: i32, mut expected2: i32) {
    let g = BlockGraph::new();
    let result1 = Arc::new(AtomicI32::new(EINPROGRESS));
    let result2 = Arc::new(AtomicI32::new(EINPROGRESS));
    let txn = JobTxn::new();
    let job1 = test_block_job_start(&g, 1, true, expected1, &result1, &txn);
    let job2 = test_block_job_start(&g, 2, true, expected2, &result2, &txn);
    job1.start();
    job2.start();
    // Release our reference now to trigger as many nice use-after-free bugs as possible.
    drop(txn);
    {
        let mut jl = job_lock();
        if expected1 == -ECANCELED {
            job1.cancel_locked(&mut jl, false);
        }
        if expected2 == -ECANCELED {
            job2.cancel_locked(&mut jl, false);
        }
    }
    wait_results(&[&result1, &result2]);
    // Failure or cancellation of one job cancels the other job
    if expected1 != 0 {
        expected2 = -ECANCELED;
    } else if expected2 != 0 {
        expected1 = -ECANCELED;
    }
    assert_eq!(result1.load(Ordering::SeqCst), expected1);
    assert_eq!(result2.load(Ordering::SeqCst), expected2);
}

#[test]
fn pair_success() {
    let _s = serial();
    pair_jobs(0, 0);
}

#[test]
fn pair_failure() {
    let _s = serial();
    // Test both orderings. The two jobs run for a different number of iterations so the
    // code path is different depending on which job fails first.
    pair_jobs(EIO, 0);
    pair_jobs(0, EIO);
}

#[test]
fn pair_cancel() {
    let _s = serial();
    pair_jobs(-ECANCELED, 0);
    pair_jobs(0, -ECANCELED);
}

#[test]
fn pair_fail_cancel_race() {
    let _s = serial();
    let g = BlockGraph::new();
    let result1 = Arc::new(AtomicI32::new(EINPROGRESS));
    let result2 = Arc::new(AtomicI32::new(EINPROGRESS));
    let txn = JobTxn::new();
    let job1 = test_block_job_start(&g, 1, true, -ECANCELED, &result1, &txn);
    let job2 = test_block_job_start(&g, 2, false, 0, &result2, &txn);
    job1.start();
    job2.start();
    {
        let mut jl = job_lock();
        job1.cancel_locked(&mut jl, false);
    }
    // Now make job2 finish before the main loop kicks jobs. This simulates the race between
    // a pending kick and another job completing.
    job2.enter();
    job2.enter();
    wait_results(&[&result1, &result2]);
    assert_eq!(result1.load(Ordering::SeqCst), -ECANCELED);
    assert_eq!(result2.load(Ordering::SeqCst), -ECANCELED);
}
