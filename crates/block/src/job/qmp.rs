// SPDX-License-Identifier: GPL-2.0-or-later

//! The job commands of job-qmp.c (`job-*`, `query-jobs`) and blockdev.c (`block-job-*`,
//! `query-block-jobs`). Each takes the BQL and then the job lock, as a monitor command runs.

use std::sync::Arc;

use ruvm_base::{Error, Result};
use ruvm_qapi::types::{BlockJobChangeOptions, BlockJobInfo, JobInfo};

use super::block_job::{find_block_job_locked, is_block_job};
use super::core::{Job, JobLock, job_get_locked, job_lock};
use super::main_loop::bql_lock;
use crate::graph::BlockGraph;

/// `find_job_locked()`.
fn find_job_locked(jl: &mut JobLock, id: &str) -> Result<Arc<Job>> {
    job_get_locked(jl, id).ok_or_else(|| Error::generic("Job not found"))
}

/// Runs `f` on job `id` with the BQL and the job lock held.
fn with_job<R>(id: &str, f: impl FnOnce(&mut JobLock, &Arc<Job>) -> Result<R>) -> Result<R> {
    let _bql = bql_lock();
    let mut jl = job_lock();
    let job = find_job_locked(&mut jl, id)?;
    f(&mut jl, &job)
}

/// Runs `f` on block job `id` with the BQL and the job lock held.
fn with_block_job<R>(id: &str, f: impl FnOnce(&mut JobLock, &Arc<Job>) -> Result<R>) -> Result<R> {
    let _bql = bql_lock();
    let mut jl = job_lock();
    let job = find_block_job_locked(&mut jl, id)?;
    f(&mut jl, &job)
}

impl BlockGraph {
    /// `job-cancel`.
    pub fn job_cancel(&self, id: &str) -> Result<()> {
        with_job(id, |jl, j| j.user_cancel_locked(jl, true))
    }

    /// `job-pause`.
    pub fn job_pause(&self, id: &str) -> Result<()> {
        with_job(id, |jl, j| j.user_pause_locked(jl))
    }

    /// `job-resume`.
    pub fn job_resume(&self, id: &str) -> Result<()> {
        with_job(id, |jl, j| j.user_resume_locked(jl))
    }

    /// `job-complete`.
    pub fn job_complete(&self, id: &str) -> Result<()> {
        with_job(id, |jl, j| j.complete_locked(jl))
    }

    /// `job-finalize`.
    pub fn job_finalize(&self, id: &str) -> Result<()> {
        with_job(id, |jl, j| {
            j.ref_locked(jl);
            let r = j.finalize_locked(jl);
            j.unref_locked(jl);
            r
        })
    }

    /// `job-dismiss`.
    pub fn job_dismiss(&self, id: &str) -> Result<()> {
        with_job(id, |jl, j| j.dismiss_locked(jl))
    }

    /// `query-jobs`: every job with an id, newest first.
    pub fn query_jobs(&self) -> Vec<JobInfo> {
        let mut jl = job_lock();
        let jobs = jl.jobs();
        jobs.iter().filter(|j| !j.is_internal()).map(|j| j.query_locked(&mut jl)).collect()
    }

    /// `block-job-set-speed`.
    pub fn block_job_set_speed(&self, device: &str, speed: i64) -> Result<()> {
        with_block_job(device, |jl, j| j.set_speed_locked(jl, speed))
    }

    /// `block-job-cancel`.
    pub fn block_job_cancel(&self, device: &str, force: Option<bool>) -> Result<()> {
        with_block_job(device, |jl, j| {
            let force = force.unwrap_or(false);
            if j.user_paused_locked(jl) && !force {
                return Err(Error::generic(format!(
                    "The block job for device '{device}' is currently paused"
                )));
            }
            j.user_cancel_locked(jl, force)
        })
    }

    /// `block-job-pause`.
    pub fn block_job_pause(&self, device: &str) -> Result<()> {
        with_block_job(device, |jl, j| j.user_pause_locked(jl))
    }

    /// `block-job-resume`.
    pub fn block_job_resume(&self, device: &str) -> Result<()> {
        with_block_job(device, |jl, j| j.user_resume_locked(jl))
    }

    /// `block-job-complete`.
    pub fn block_job_complete(&self, device: &str) -> Result<()> {
        with_block_job(device, |jl, j| j.complete_locked(jl))
    }

    /// `block-job-finalize`.
    pub fn block_job_finalize(&self, id: &str) -> Result<()> {
        with_block_job(id, |jl, j| {
            j.ref_locked(jl);
            let r = j.finalize_locked(jl);
            j.unref_locked(jl);
            r
        })
    }

    /// `block-job-dismiss`.
    pub fn block_job_dismiss(&self, id: &str) -> Result<()> {
        with_block_job(id, |jl, j| j.dismiss_locked(jl))
    }

    /// `block-job-change`.
    pub fn block_job_change(&self, opts: &BlockJobChangeOptions) -> Result<()> {
        with_block_job(&opts.id, |jl, j| j.change_locked(jl, opts))
    }

    /// `query-block-jobs`: every block job with an id, newest first.
    pub fn query_block_jobs(&self) -> Result<Vec<BlockJobInfo>> {
        let mut jl = job_lock();
        let jobs = jl.jobs();
        let mut out = Vec::new();
        for j in jobs.iter().filter(|j| is_block_job(j) && !j.is_internal()) {
            out.push(j.block_query_locked(&mut jl)?);
        }
        Ok(out)
    }
}
