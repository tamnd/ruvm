// SPDX-License-Identifier: GPL-2.0-or-later

//! `blockdev-create` and `x-blockdev-amend` from block/create.c and block/amend.c: jobs that
//! run the `.bdrv_co_create` and `.bdrv_co_amend` callbacks of a format driver. The jobs do
//! all of their work in one step and report 0 of 1 then 1 of 1 as their progress.
//!
//! Differences from QEMU:
//!
//! - The create job runs on a thread of its own, so it holds on to the graph: the QMP entry
//!   point takes an `Arc<BlockGraph>`.
//! - A driver cannot tell whether it amends before it is asked to, so the `Driver does not
//!   support x-blockdev-amend` error comes from the job instead of the command. QEMU cannot
//!   reach that error either: the schema only has amend options for drivers that amend.
//! - The `.bdrv_amend_pre_run` and `.bdrv_amend_clean` hooks of LUKS run inside
//!   [`Driver::amend`](crate::node::Driver::amend), so the job has no hooks of its own.
//! - There is no driver whitelist, so `Driver is not whitelisted` does not happen.

use std::any::Any;
use std::io;
use std::sync::{Arc, Mutex};

use ruvm_base::{Error, Result};
use ruvm_qapi::types::{BlockdevAmendOptions, BlockdevCreateOptions, JobType};

use super::core::{JOB_DEFAULT, JOB_MANUAL_DISMISS, Job, JobDriver, JobErr};
use super::main_loop::bql_lock;
use crate::drivers;
use crate::graph::BlockGraph;
use crate::node::Node;

/// The errno of a failed callback: the one of its host error if it has one, else `EINVAL`.
fn errno_of(e: &Error) -> i32 {
    std::error::Error::source(e)
        .and_then(|s| s.downcast_ref::<io::Error>())
        .and_then(io::Error::raw_os_error)
        .unwrap_or(libc::EINVAL)
}

/// `BlockdevCreateJob`.
struct CreateJob {
    graph: Arc<BlockGraph>,
    drv: &'static drivers::DriverDef,
    opts: Mutex<Option<BlockdevCreateOptions>>,
}

impl JobDriver for CreateJob {
    /// `blockdev_create_run()`.
    fn run(&self, job: &Arc<Job>) -> Result<(), JobErr> {
        job.progress_set_remaining(1);
        let opts = self.opts.lock().unwrap().take().expect("create options run once");
        let create = self.drv.create.expect("checked when the job was created");
        let r = create(&self.graph, opts.u);
        job.progress_update(1);
        r.map_err(|e| JobErr::with(errno_of(&e), e))
    }

    fn as_any(&self) -> Option<&dyn Any> {
        Some(self)
    }
}

/// `BlockdevAmendJob`.
struct AmendJob {
    opts: BlockdevAmendOptions,
    bs: Arc<Node>,
    force: bool,
}

impl JobDriver for AmendJob {
    /// `blockdev_amend_run()`.
    fn run(&self, job: &Arc<Job>) -> Result<(), JobErr> {
        let _g = crate::graph_lock::rdlock();
        job.progress_set_remaining(1);
        let r = match self.bs.driver.amend(&self.bs, &self.opts, self.force) {
            Some(r) => r,
            None => Err(Error::generic("Driver does not support x-blockdev-amend")),
        };
        job.progress_update(1);
        r.map_err(|e| JobErr::with(errno_of(&e), e))
    }

    fn as_any(&self) -> Option<&dyn Any> {
        Some(self)
    }
}

impl BlockGraph {
    /// `qmp_blockdev_create()`: starts the job `job_id` that creates an image as `options`
    /// say. The job waits in CONCLUDED for `job-dismiss`. [`BlockGraph::blockdev_create`]
    /// does the same without a job.
    pub fn blockdev_create_job(
        self: &Arc<Self>,
        job_id: &str,
        options: BlockdevCreateOptions,
    ) -> Result<()> {
        let _bql = bql_lock();
        let fmt = options.u.tag().as_str();
        let Some(drv) = drivers::find_format(fmt) else {
            return Err(Error::generic(format!("Block driver '{fmt}' not found or not supported")));
        };
        // Error out if the driver doesn't support .bdrv_co_create
        if drv.create.is_none() {
            return Err(Error::generic("Driver does not support blockdev-create"));
        }
        // Create the block job
        let job = Job::create(
            Some(job_id),
            JobType::Create,
            None,
            JOB_DEFAULT | JOB_MANUAL_DISMISS,
            None,
            None,
        )?;
        job.set_driver(Box::new(CreateJob {
            graph: self.clone(),
            drv,
            opts: Mutex::new(Some(options)),
        }));
        job.start();
        Ok(())
    }

    /// `qmp_x_blockdev_amend()`: starts the job `job_id` that changes the options of the
    /// node `node_name` in place. The job waits in CONCLUDED for `job-dismiss`.
    /// [`BlockGraph::x_blockdev_amend`] does the same without a job.
    pub fn x_blockdev_amend_job(
        &self,
        job_id: &str,
        node_name: &str,
        options: BlockdevAmendOptions,
        force: Option<bool>,
    ) -> Result<()> {
        let _bql = bql_lock();
        let fmt = options.u.tag().as_str();
        let drv = drivers::find_format(fmt);
        let bs = {
            let _g = crate::graph_lock::rdlock();
            // bdrv_lookup_bs(NULL, node_name, errp)
            let Some(bs) = self.find_node(node_name) else {
                return Err(Error::generic(format!(
                    "Cannot find device='' nor node-name='{node_name}'"
                )));
            };
            let Some(drv) = drv else {
                return Err(Error::generic(format!(
                    "Block driver '{fmt}' not found or not supported"
                )));
            };
            if bs.driver_name != drv.format_name {
                return Err(Error::generic(
                    "x-blockdev-amend doesn't support changing the block driver",
                ));
            }
            bs
        };
        // Create the block job
        let job = Job::create(
            Some(job_id),
            JobType::Amend,
            None,
            JOB_DEFAULT | JOB_MANUAL_DISMISS,
            None,
            None,
        )?;
        job.set_driver(Box::new(AmendJob { opts: options, bs, force: force.unwrap_or(false) }));
        job.start();
        Ok(())
    }
}
