// SPDX-License-Identifier: GPL-2.0-or-later

//! Test helpers for the job tests, and the ports of the job unit tests of QEMU.
//!
//! The job list is global, as in QEMU, so the job tests run one at a time ([`serial`]) and
//! use job ids no other test uses.

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use ruvm_qapi::json;
use ruvm_qapi::types::{BlockdevOptions, JobStatus};
use ruvm_qapi::visit::{QObjectInputVisitor, Visit};

use super::core::{Job, job_lock};
use crate::backend::BlockBackend;
use crate::event::BlockEvent;
use crate::graph::BlockGraph;
use crate::perm::BLK_PERM_ALL;

mod backup;
mod blockjob;
mod commit;
mod cow;
mod create;
mod drain;
mod mirror;
mod stream;
mod txn;

static EVENTS: Mutex<Vec<BlockEvent>> = Mutex::new(Vec::new());
static SERIAL: Mutex<()> = Mutex::new(());

/// Records a job event.
pub(crate) fn record_event(ev: &BlockEvent) {
    EVENTS.lock().unwrap().push(ev.clone());
}

/// Takes the recorded events that concern the job or device `id`.
pub(crate) fn take_events(id: &str) -> Vec<BlockEvent> {
    let mut all = EVENTS.lock().unwrap();
    let mut out = Vec::new();
    all.retain(|e| {
        let mine = match e {
            BlockEvent::JobStatusChange { id: i, .. }
            | BlockEvent::BlockJobPending { id: i, .. } => i == id,
            BlockEvent::BlockJobCompleted { device, .. }
            | BlockEvent::BlockJobCancelled { device, .. }
            | BlockEvent::BlockJobReady { device, .. }
            | BlockEvent::BlockJobError { device, .. } => device == id,
            _ => false,
        };
        if mine {
            out.push(e.clone());
        }
        !mine
    });
    out
}

/// The status changes among `evs`.
pub(crate) fn statuses(evs: &[BlockEvent]) -> Vec<JobStatus> {
    evs.iter()
        .filter_map(|e| match e {
            BlockEvent::JobStatusChange { status, .. } => Some(*status),
            _ => None,
        })
        .collect()
}

/// Held by a job test while it runs.
pub(crate) struct Serial {
    _owner: crate::drain::DrainAllOwner,
    _g: MutexGuard<'static, ()>,
}

/// Runs the job tests one at a time, and keeps the drain-all sections of other tests out, as
/// they would pause the jobs and add status changes.
pub(crate) fn serial() -> Serial {
    let g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    Serial { _owner: crate::drain::DrainAllOwner::acquire(), _g: g }
}

/// Adds a node from JSON.
pub(crate) fn add(g: &BlockGraph, s: &str) {
    let mut v = QObjectInputVisitor::new(json::from_str(s).unwrap());
    let mut o = BlockdevOptions::default();
    BlockdevOptions::visit(&mut v, None, &mut o).unwrap();
    g.blockdev_add(o).unwrap();
}

/// `create_blk()`: a backend on a null-co node, named `name` in the monitor if given.
pub(crate) fn create_blk(g: &BlockGraph, node: &str, name: Option<&str>) -> Arc<BlockBackend> {
    add(g, &format!(r#"{{"driver": "null-co", "node-name": "{node}", "read-zeroes": true}}"#));
    let bs = g.find_node(node).unwrap();
    let blk =
        Arc::new(BlockBackend::with_node(name.map(str::to_string), bs, 0, BLK_PERM_ALL).unwrap());
    if let Some(n) = name {
        g.monitor_add_blk(n, blk.clone()).unwrap();
    }
    blk
}

/// Waits until `f` holds, failing after ten seconds.
pub(crate) fn wait_for(what: &str, mut f: impl FnMut() -> bool) {
    let end = Instant::now() + Duration::from_secs(10);
    while !f() {
        assert!(Instant::now() < end, "timed out waiting for {what}");
        crate::job::main_loop::poll_bhs();
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// Waits until the job is in `status`.
pub(crate) fn wait_status(job: &Job, status: JobStatus) {
    wait_for(status.as_str(), || job.status() == status);
}

/// `assert_job_status_is()`.
pub(crate) fn assert_status(job: &Job, status: JobStatus) {
    let _jl = job_lock();
    assert_eq!(job.s().status, status);
}
