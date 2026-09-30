// SPDX-License-Identifier: GPL-2.0-or-later

//! `BlockJob` from blockjob.c: a job that works on block nodes. It holds the nodes it uses
//! through edges of its own (the `child_job` class), blocks every operation on them, limits
//! its speed and reports I/O errors.

use std::sync::{Arc, Mutex, Weak};

use ruvm_base::{Error, ErrorClass, Result};
use ruvm_qapi::types::{
    BlockDeviceIoStatus, BlockErrorAction, BlockJobChangeOptions, BlockJobInfo, BlockJobInfoU,
    BlockdevOnError, IoOperationType, JobType, JobVerb,
};

use super::blocker::{Reason, op_block_all, op_unblock_all};
use super::core::{Job, JobCb, JobLock, JobTxn, job_get_locked, job_lock, strerror};
use super::ratelimit::RateLimit;
use crate::event::BlockEvent;
use crate::graph::BlockGraph;
use crate::node::{Node, Parent, ParentOps, new_edge_id};

/// `BLOCK_JOB_SLICE_TIME`: 100 ms.
pub(crate) const BLOCK_JOB_SLICE_TIME: u64 = 100_000_000;

/// The block job part of a job.
pub(crate) struct BlockJob {
    /// `job->nodes`: the nodes and the edge ids of the job's edges on them, newest first.
    nodes: Mutex<Vec<(Arc<Node>, u64)>>,
    /// `job->blocker`.
    blocker: Arc<Reason>,
    /// `job->limit`.
    pub(crate) limit: RateLimit,
}

impl BlockJob {
    /// A block job part for a job of `job_type`.
    pub(crate) fn new(job_type: JobType) -> BlockJob {
        BlockJob {
            nodes: Mutex::new(Vec::new()),
            blocker: Arc::new(Reason(format!(
                "block device is in use by block job: {}",
                job_type.as_str()
            ))),
            limit: RateLimit::default(),
        }
    }

    /// `block_job_free()`.
    pub(crate) fn free(&self, _job: &Arc<Job>) {
        self.remove_all_bdrv();
    }

    /// `block_job_remove_all_bdrv()`.
    pub(crate) fn remove_all_bdrv(&self) {
        loop {
            let Some((bs, edge)) = ({
                let mut n = self.nodes.lock().unwrap();
                if n.is_empty() { None } else { Some(n.remove(0)) }
            }) else {
                return;
            };
            op_unblock_all(&bs, &self.blocker);
            let _d = bs.drained();
            let _ = bs.update_parent(edge, None);
        }
    }

    /// The nodes the job holds.
    #[allow(dead_code, reason = "for tests")]
    pub(crate) fn nodes(&self) -> Vec<Arc<Node>> {
        self.nodes.lock().unwrap().iter().map(|(n, _)| n.clone()).collect()
    }
}

/// `child_job`: the edge from a block job to a node it uses.
struct ChildJob(Weak<Job>);

impl ParentOps for ChildJob {
    fn drained_begin(&self) {
        if let Some(j) = self.0.upgrade() {
            j.pause();
        }
    }

    fn drained_end(&self) {
        if let Some(j) = self.0.upgrade() {
            j.resume();
        }
    }

    fn drained_poll(&self) -> bool {
        let Some(job) = self.0.upgrade() else {
            return false;
        };
        // An inactive or completed job doesn't have any pending requests. Jobs with
        // !job->busy are either already paused or have a pause point after being reentered,
        // so no job driver code will run before they pause.
        {
            let _jl = job_lock();
            let s = job.s();
            if !s.busy || Job::is_completed_state(&s) {
                return false;
            }
        }
        // Otherwise, assume that it isn't fully stopped yet, but allow the job to override
        // this assumption.
        match job.driver_opt() {
            Some(d) => d.drained_poll(&job),
            None => true,
        }
    }

    fn stay_at_node(&self) -> bool {
        true
    }
}

/// `is_block_job()`.
pub(crate) fn is_block_job(job: &Job) -> bool {
    matches!(job.job_type(), JobType::Backup | JobType::Commit | JobType::Mirror | JobType::Stream)
}

/// `block_job_get_locked()`.
pub(crate) fn block_job_get_locked(jl: &mut JobLock, id: &str) -> Option<Arc<Job>> {
    job_get_locked(jl, id).filter(|j| is_block_job(j))
}

/// `bdrv_get_device_name()`: the name of the monitor's block backend on `bs`, or an empty
/// string.
pub(crate) fn device_name(graph: &BlockGraph, bs: &Arc<Node>) -> String {
    let backends = graph.backends.lock().unwrap();
    for (name, blk) in backends.iter() {
        if blk.root().is_some_and(|r| Arc::ptr_eq(&r, bs)) {
            return name.clone();
        }
    }
    String::new()
}

/// `bdrv_get_device_or_node_name()`.
pub(crate) fn device_or_node_name(graph: &BlockGraph, bs: &Arc<Node>) -> String {
    let d = device_name(graph, bs);
    if d.is_empty() { bs.name.clone() } else { d }
}

/// What `block_job_create()` gets besides the node.
pub(crate) struct BlockJobParams<'a> {
    pub job_id: Option<&'a str>,
    pub job_type: JobType,
    pub txn: Option<Arc<JobTxn>>,
    pub perm: u64,
    pub shared: u64,
    pub speed: i64,
    pub flags: u32,
    pub cb: Option<JobCb>,
}

/// `block_job_create()`: a block job on `bs`, which it holds as its "main node". The caller
/// installs the driver with [`Job::set_driver`] before it starts the job.
pub(crate) fn block_job_create(
    graph: &BlockGraph,
    bs: &Arc<Node>,
    p: BlockJobParams<'_>,
) -> Result<Arc<Job>> {
    let dev;
    let mut job_id = p.job_id;
    if job_id.is_none() && p.flags & super::core::JOB_INTERNAL == 0 {
        dev = device_name(graph, bs);
        job_id = Some(&dev);
    }
    let job =
        Job::create(job_id, p.job_type, p.txn, p.flags, p.cb, Some(BlockJob::new(p.job_type)))?;
    let r = block_job_add_bdrv(&job, "main node", bs, p.perm, p.shared)
        .and_then(|()| job.set_speed(p.speed));
    if let Err(e) = r {
        job.early_fail();
        return Err(e);
    }
    Ok(job)
}

/// `block_job_add_bdrv()`: the job holds `bs` with `perm`, sharing `shared`, and blocks every
/// operation on it.
pub(crate) fn block_job_add_bdrv(
    job: &Arc<Job>,
    name: &str,
    bs: &Arc<Node>,
    perm: u64,
    shared: u64,
) -> Result<()> {
    let bj = job.bj();
    let edge = new_edge_id();
    let parent = Parent {
        id: edge,
        // child_job_get_parent_desc()
        desc: format!("{} job '{}'", job.job_type().as_str(), job.id_str()),
        child_name: name.to_string(),
        perm,
        shared,
        ops: Some(Arc::new(ChildJob(Arc::downgrade(job)))),
        quiesced: false,
    };
    bs.update_parent(edge, Some(parent))?;
    bj.nodes.lock().unwrap().insert(0, (bs.clone(), edge));
    op_block_all(bs, &bj.blocker);
    Ok(())
}

impl Job {
    /// The block job part.
    pub(crate) fn bj(&self) -> &BlockJob {
        self.bj.as_ref().expect("a block job")
    }

    /// `block_job_set_speed_locked()`.
    pub(crate) fn set_speed_locked(&self, jl: &mut JobLock, speed: i64) -> Result<()> {
        let old_speed = self.s().speed;
        self.apply_verb_locked(jl, JobVerb::SetSpeed)?;
        if speed < 0 {
            return Err(Error::generic("Parameter 'speed' expects a non-negative value"));
        }
        self.bj().limit.set_speed(speed as u64, BLOCK_JOB_SLICE_TIME);
        self.s().speed = speed;
        if let Some(d) = self.driver_opt() {
            let me = self.self_arc();
            jl.unlocked(|| d.set_speed(&me, speed));
        }
        if speed != 0 && speed <= old_speed {
            return Ok(());
        }
        // kick only if a timer is pending
        self.enter_cond_locked(jl, Some(|s| s.timer_pending()));
        Ok(())
    }

    /// `block_job_set_speed()`.
    pub(crate) fn set_speed(&self, speed: i64) -> Result<()> {
        let mut jl = job_lock();
        self.set_speed_locked(&mut jl, speed)
    }

    /// `block_job_change_locked()`.
    pub(crate) fn change_locked(
        &self,
        jl: &mut JobLock,
        opts: &BlockJobChangeOptions,
    ) -> Result<()> {
        self.apply_verb_locked(jl, JobVerb::Change)?;
        let me = self.self_arc();
        match self.driver_opt() {
            Some(d) => jl.unlocked(|| d.change(&me, opts)),
            None => None,
        }
        .unwrap_or_else(|| Err(Error::generic("Job type does not support change")))
    }

    /// `block_job_ratelimit_processed_bytes()`.
    pub(crate) fn ratelimit_processed_bytes(&self, n: u64) {
        self.bj().limit.calculate_delay(n);
    }

    /// `block_job_ratelimit_sleep()`.
    pub(crate) fn ratelimit_sleep(&self) {
        // Sleep at least once. If the job is reentered early, keep waiting until we've
        // waited for the full time that is necessary to keep the job at the right speed.
        //
        // Make sure to recalculate the delay after each (possibly interrupted) sleep because
        // the speed can change while the job has yielded.
        loop {
            let delay_ns = self.bj().limit.calculate_delay(0);
            self.sleep_ns(delay_ns);
            if delay_ns == 0 || self.is_cancelled() {
                break;
            }
        }
    }

    /// `block_job_query_locked()`.
    pub(crate) fn block_query_locked(&self, jl: &mut JobLock) -> Result<BlockJobInfo> {
        if self.is_internal() {
            return Err(Error::generic("Cannot query QEMU internal jobs"));
        }
        let mut info = {
            let s = self.s();
            BlockJobInfo {
                device: self.id_str().to_string(),
                len: s.progress.total as i64,
                offset: s.progress.current as i64,
                busy: s.busy,
                paused: s.pause_count > 0,
                speed: s.speed,
                io_status: s.iostatus,
                ready: Job::is_ready_state(&s),
                status: s.status,
                auto_finalize: s.auto_finalize,
                auto_dismiss: s.auto_dismiss,
                error: if s.ret != 0 {
                    Some(match &s.err {
                        Some(e) => e.message().to_string(),
                        None => strerror(-s.ret),
                    })
                } else {
                    None
                },
                u: BlockJobInfoU::for_tag(self.job_type()),
            }
        };
        if let Some(d) = self.driver_opt() {
            let me = self.self_arc();
            jl.unlocked(|| d.query(&me, &mut info));
        }
        Ok(info)
    }

    /// `block_job_error_action()`: what to do about the I/O error `error` (a positive errno)
    /// under the policy `on_err`. For `stop`, the job pauses as if the user paused it.
    pub(crate) fn error_action(
        &self,
        on_err: BlockdevOnError,
        is_read: bool,
        error: i32,
    ) -> BlockErrorAction {
        let action = match on_err {
            BlockdevOnError::Enospc | BlockdevOnError::Auto => {
                if error == libc::ENOSPC {
                    BlockErrorAction::Stop
                } else {
                    BlockErrorAction::Report
                }
            }
            BlockdevOnError::Stop => BlockErrorAction::Stop,
            BlockdevOnError::Report => BlockErrorAction::Report,
            BlockdevOnError::Ignore => BlockErrorAction::Ignore,
        };
        if !self.is_internal() {
            super::send_event(BlockEvent::BlockJobError {
                device: self.id_str().to_string(),
                operation: if is_read { IoOperationType::Read } else { IoOperationType::Write },
                action,
            });
        }
        if action == BlockErrorAction::Stop {
            let mut jl = job_lock();
            if !self.s().user_paused {
                self.pause_locked(&mut jl);
                // make the pause user visible, which will be resumed from QMP.
                self.s().user_paused = true;
            }
            // block_job_iostatus_set_err_locked()
            let mut s = self.s();
            if s.iostatus == BlockDeviceIoStatus::Ok {
                s.iostatus = if error == libc::ENOSPC {
                    BlockDeviceIoStatus::Nospace
                } else {
                    BlockDeviceIoStatus::Failed
                };
            }
        }
        action
    }

    fn progress_speed(&self) -> (u64, u64, u64) {
        let s = self.s();
        (s.progress.total, s.progress.current, s.speed.max(0) as u64)
    }

    /// `block_job_event_cancelled_locked()`.
    pub(crate) fn event_cancelled_locked(&self, _jl: &mut JobLock) {
        if self.bj.is_none() || self.is_internal() {
            return;
        }
        let (len, offset, speed) = self.progress_speed();
        super::send_event(BlockEvent::BlockJobCancelled {
            job_type: self.job_type(),
            device: self.id_str().to_string(),
            len,
            offset,
            speed,
        });
    }

    /// `block_job_event_completed_locked()`.
    pub(crate) fn event_completed_locked(&self, _jl: &mut JobLock) {
        if self.bj.is_none() || self.is_internal() {
            return;
        }
        let error = {
            let s = self.s();
            if s.ret < 0 { s.err.as_ref().map(|e| e.message().to_string()) } else { None }
        };
        let (len, offset, speed) = self.progress_speed();
        super::send_event(BlockEvent::BlockJobCompleted {
            job_type: self.job_type(),
            device: self.id_str().to_string(),
            len,
            offset,
            speed,
            error,
        });
    }

    /// `block_job_event_pending_locked()`.
    pub(crate) fn event_pending_locked(&self, _jl: &mut JobLock) {
        if self.bj.is_none() || self.is_internal() {
            return;
        }
        super::send_event(BlockEvent::BlockJobPending {
            job_type: self.job_type(),
            id: self.id_str().to_string(),
        });
    }

    /// `block_job_event_ready_locked()`.
    pub(crate) fn event_ready_locked(&self, _jl: &mut JobLock) {
        if self.bj.is_none() || self.is_internal() {
            return;
        }
        let (len, offset, speed) = self.progress_speed();
        super::send_event(BlockEvent::BlockJobReady {
            job_type: self.job_type(),
            device: self.id_str().to_string(),
            len,
            offset,
            speed,
        });
    }
}

/// `find_block_job_locked()` of blockdev.c.
pub(crate) fn find_block_job_locked(jl: &mut JobLock, id: &str) -> Result<Arc<Job>> {
    block_job_get_locked(jl, id).ok_or_else(|| {
        Error::new(ErrorClass::DeviceNotActive, format!("Block job '{id}' not found"))
    })
}
