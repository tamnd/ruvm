// SPDX-License-Identifier: GPL-2.0-or-later

//! The block job parts of tests/unit/test-bdrv-drain.c, and tests/unit/test-block-backend.c.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::time::Duration;

use ruvm_base::Result;
use ruvm_qapi::types::JobType;

use super::cow::cow_node;
use super::*;
use crate::drain::{drain_all_begin, drain_all_end};
use crate::job::block_job::{BlockJobParams, block_job_add_bdrv, block_job_create};
use crate::job::core::{JOB_DEFAULT, JobDriver, JobErr};
use crate::job::main_loop::bql_lock;
use crate::node::{ENOMEDIUM, Node};

/// `TestBlockJob` of test-bdrv-drain.c.
struct TestBlockJob {
    bs: Arc<Node>,
    run_ret: i32,
    prepare_ret: i32,
    running: AtomicBool,
    should_complete: AtomicBool,
}

impl JobDriver for TestBlockJob {
    fn run(&self, job: &Arc<Job>) -> Result<(), JobErr> {
        // We are running the actual job code past the pause point in job_co_entry().
        self.running.store(true, Ordering::SeqCst);
        job.transition_to_ready();
        while !self.should_complete.load(Ordering::SeqCst) {
            // Avoid job_sleep_ns() because it marks the job as !busy. We want to emulate
            // some actual activity (probably some I/O) here so that drain has to wait for
            // this activity to stop.
            std::thread::sleep(Duration::from_millis(1));
            job.pause_point();
        }
        if self.run_ret == 0 { Ok(()) } else { Err(JobErr::errno(-self.run_ret)) }
    }

    fn has_complete(&self) -> bool {
        true
    }

    fn complete(&self, _job: &Arc<Job>) -> Result<()> {
        self.should_complete.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn prepare(&self, _job: &Arc<Job>) -> Option<Result<(), JobErr>> {
        // Provoke an AIO_WAIT_WHILE() call to verify there is no deadlock
        let _ = self.bs.flush();
        Some(if self.prepare_ret == 0 { Ok(()) } else { Err(JobErr::errno(-self.prepare_ret)) })
    }

    fn commit(&self, _job: &Arc<Job>) {
        let _ = self.bs.flush();
    }

    fn abort(&self, _job: &Arc<Job>) {
        let _ = self.bs.flush();
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DrainType {
    All,
    Node,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum JobResult {
    Success,
    FailRun,
    FailPrepare,
}

fn drain_begin(t: DrainType, bs: &Node) {
    match t {
        DrainType::All => drain_all_begin(),
        DrainType::Node => bs.drained_begin(),
    }
}

fn drain_end(t: DrainType, bs: &Node) {
    match t {
        DrainType::All => drain_all_end(),
        DrainType::Node => bs.drained_end(),
    }
}

fn check_paused(job: &Job, t: DrainType) {
    let _jl = job_lock();
    let s = job.s();
    // bdrv_drain_all() drains both src and target
    assert_eq!(s.pause_count, if t == DrainType::All { 2 } else { 1 });
    assert!(s.paused);
    assert!(!s.busy); // The job is paused
}

fn check_running(job: &Job) {
    // paused is reset on the job's thread, wait for it
    wait_for("the job to resume", || !job.with_state(|s| s.paused));
    let _jl = job_lock();
    let s = job.s();
    assert_eq!(s.pause_count, 0);
    assert!(s.busy); // We're in qemu_co_sleep_ns()
}

/// `test_blockjob_common_drain_node()`.
fn blockjob_common_drain_node(t: DrainType, result: JobResult, drain_child: bool) {
    let (src, _) = cow_node("source", 65536);
    let (src_backing, _) = cow_node("source-backing", 65536);
    let (src_overlay, _) = cow_node("source-overlay", 65536);
    src_overlay.set_backing_hd(Some(src.clone())).unwrap();
    src.set_backing_hd(Some(src_backing.clone())).unwrap();
    let blk_src = BlockBackend::new_empty(None, BLK_PERM_ALL, BLK_PERM_ALL, false);
    blk_src.insert(src_overlay.clone()).unwrap();
    let drain_bs = if drain_child { &src_backing } else { &src };

    let (target, _) = cow_node("target", 65536);
    let blk_target = BlockBackend::new_empty(None, BLK_PERM_ALL, BLK_PERM_ALL, false);
    blk_target.insert(target.clone()).unwrap();

    let g = BlockGraph::new();
    let job = block_job_create(
        &g,
        &src,
        BlockJobParams {
            job_id: Some("job0"),
            job_type: JobType::Commit,
            txn: None,
            perm: 0,
            shared: BLK_PERM_ALL,
            speed: 0,
            flags: JOB_DEFAULT,
            cb: None,
        },
    )
    .unwrap();
    block_job_add_bdrv(&job, "target", &target, 0, BLK_PERM_ALL).unwrap();
    job.set_driver(Box::new(TestBlockJob {
        bs: src.clone(),
        run_ret: if result == JobResult::FailRun { -libc::EIO } else { 0 },
        prepare_ret: if result == JobResult::FailPrepare { -libc::EIO } else { 0 },
        running: AtomicBool::new(false),
        should_complete: AtomicBool::new(false),
    }));

    job.start();
    // job_co_entry() runs on the job's thread, wait for the actual job code to start (we
    // don't want to catch the job in the pause point in job_co_entry().
    wait_for("the job to run", || job.is_ready());
    check_running(&job);

    drain_begin(t, drain_bs);
    check_paused(&job, t);
    drain_end(t, drain_bs);
    check_running(&job);

    drain_begin(t, &target);
    check_paused(&job, t);
    drain_end(t, &target);
    check_running(&job);

    let ret = {
        let _bql = bql_lock();
        let mut jl = job_lock();
        job.complete_sync_locked(&mut jl).unwrap()
    };
    assert_eq!(ret, if result == JobResult::Success { 0 } else { -libc::EIO });
    // The job is gone, and so are its edges.
    assert_eq!(src.parent_count(), 1);
    assert_eq!(target.parent_count(), 1);
}

fn blockjob_common(t: DrainType, result: JobResult) {
    blockjob_common_drain_node(t, result, false);
    blockjob_common_drain_node(t, result, true);
}

#[test]
fn blockjob_drain_all() {
    let _s = serial();
    blockjob_common(DrainType::All, JobResult::Success);
}

#[test]
fn blockjob_drain() {
    let _s = serial();
    blockjob_common(DrainType::Node, JobResult::Success);
}

#[test]
fn blockjob_error_drain_all() {
    let _s = serial();
    blockjob_common(DrainType::All, JobResult::FailRun);
    blockjob_common(DrainType::All, JobResult::FailPrepare);
}

#[test]
fn blockjob_error_drain() {
    let _s = serial();
    blockjob_common(DrainType::Node, JobResult::FailRun);
    blockjob_common(DrainType::Node, JobResult::FailPrepare);
}

/// `TestDropBackingBlockJob`.
struct DropBacking {
    did_complete: Arc<AtomicBool>,
    detach_also: Arc<Node>,
    bs: Arc<Node>,
    should_complete: Arc<AtomicBool>,
}

impl JobDriver for DropBacking {
    fn run(&self, job: &Arc<Job>) -> Result<(), JobErr> {
        while !self.should_complete.load(Ordering::SeqCst) {
            job.sleep_ns(0);
        }
        Ok(())
    }

    fn commit(&self, _job: &Arc<Job>) {
        self.bs.set_backing_hd(None).unwrap();
        self.detach_also.set_backing_hd(None).unwrap();
        self.did_complete.store(true, Ordering::SeqCst);
    }
}

/// `test_blockjob_commit_by_drained_end()`: the job's commit callback changes the graph
/// right after a drained section of the node below it ends.
///
/// QEMU asks the job to complete before the drained section, and relies on the job being
/// caught at its pause point by the drain before it sees the request. A job thread would
/// see it at once, so here the request comes inside the drained section.
#[test]
fn blockjob_commit_by_drained_end() {
    let _s = serial();
    let (bs_child, _) = cow_node("child-node", 65536);
    let mut parents = Vec::new();
    for i in 0..3 {
        let (p, _) = cow_node(&format!("parent-node-{i}"), 65536);
        p.set_backing_hd(Some(bs_child.clone())).unwrap();
        parents.push(p);
    }
    let g = BlockGraph::new();
    let did_complete = Arc::new(AtomicBool::new(false));
    let should_complete = Arc::new(AtomicBool::new(false));
    let job = block_job_create(
        &g,
        &parents[2],
        BlockJobParams {
            job_id: Some("job"),
            job_type: JobType::Commit,
            txn: None,
            perm: 0,
            shared: BLK_PERM_ALL,
            speed: 0,
            flags: JOB_DEFAULT,
            cb: None,
        },
    )
    .unwrap();
    job.set_driver(Box::new(DropBacking {
        did_complete: did_complete.clone(),
        detach_also: parents[0].clone(),
        bs: parents[2].clone(),
        should_complete: should_complete.clone(),
    }));
    let _bql = bql_lock();
    job.start();
    bs_child.drained_begin();
    should_complete.store(true, Ordering::SeqCst);
    assert!(job.with_state(|s| s.paused));
    assert!(!did_complete.load(Ordering::SeqCst));
    bs_child.drained_end();
    wait_for("the job to complete", || did_complete.load(Ordering::SeqCst));
    assert!(parents[2].backing().is_none());
    assert!(parents[0].backing().is_none());
    assert!(parents[1].backing().is_some());
}

/// test-block-backend.c: draining a backend without a medium finishes, and its requests
/// fail with `ENOMEDIUM`.
#[test]
fn drain_aio_error() {
    let blk = BlockBackend::new_empty(None, BLK_PERM_ALL, BLK_PERM_ALL, false);
    let e = blk.flush().unwrap_err();
    assert_eq!(e.raw_os_error(), Some(ENOMEDIUM));
    blk.drain();
    let _owner = crate::drain::DrainAllOwner::acquire();
    let completed = AtomicI32::new(0);
    std::thread::scope(|sc| {
        sc.spawn(|| {
            let e = blk.flush().unwrap_err();
            assert_eq!(e.raw_os_error(), Some(ENOMEDIUM));
            completed.store(1, Ordering::SeqCst);
        });
    });
    crate::drain::drain_all();
    assert_eq!(completed.load(Ordering::SeqCst), 1);
}
