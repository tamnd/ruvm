// SPDX-License-Identifier: GPL-2.0-or-later

//! The `blockdev-create` and `x-blockdev-amend` jobs, as iotests 206, 210 and 295 drive
//! them: a raw file and a LUKS image made by jobs, key slots added and refused by amend
//! jobs, and the errors of the commands and of the jobs.

// The paths go into JSON unescaped, and the messages are the Unix ones.
#![cfg(unix)]

use ruvm_qapi::types::{BlockdevAmendOptions, BlockdevCreateOptions, JobInfo, JobType};

use super::*;
use crate::node::Node;

fn setup() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        ruvm_crypto::pbkdf::set_iters_per_second_override(Some(1000));
        ruvm_crypto::secret::secret_object_add_global("secret,id=jobsec0,data=first").unwrap();
        ruvm_crypto::secret::secret_object_add_global("secret,id=jobsec1,data=second").unwrap();
    });
}

fn from_json<T: Visit + Default>(s: &str) -> T {
    let mut v = QObjectInputVisitor::new(json::from_str(s).unwrap());
    let mut o = T::default();
    T::visit(&mut v, None, &mut o).unwrap();
    o
}

fn scratch(test: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("ruvm-jobcreate-{}-{test}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn job(id: &str) -> Arc<Job> {
    let mut jl = job_lock();
    super::super::core::job_get_locked(&mut jl, id).unwrap()
}

fn info(g: &BlockGraph, id: &str) -> JobInfo {
    g.query_jobs().into_iter().find(|j| j.id == id).unwrap()
}

/// Waits for the job `id` to conclude, dismisses it and returns what `query-jobs` said of
/// it just before.
fn run_to_end(g: &BlockGraph, id: &str) -> JobInfo {
    wait_status(&job(id), JobStatus::Concluded);
    let i = info(g, id);
    g.job_dismiss(id).unwrap();
    assert!(g.query_jobs().iter().all(|j| j.id != id));
    i
}

fn slots(node: &Node) -> Vec<bool> {
    use ruvm_qapi::types::ImageInfoSpecificU;
    match node.driver.get_specific_info(node).unwrap().unwrap().u {
        ImageInfoSpecificU::Luks(w) => w.data.slots.iter().map(|s| s.active).collect(),
        _ => panic!("not luks info"),
    }
}

#[test]
fn create_job_file() {
    let _s = serial();
    let dir = scratch("file");
    let path = dir.join("a.img");
    let path = path.to_str().unwrap();
    let g = Arc::new(BlockGraph::new());
    take_events("cj0");
    g.blockdev_create_job(
        "cj0",
        from_json(&format!(r#"{{"driver": "file", "filename": "{path}", "size": 1048576}}"#)),
    )
    .unwrap();
    let i = run_to_end(&g, "cj0");
    assert_eq!(i.type_, JobType::Create);
    assert_eq!((i.current_progress, i.total_progress), (1, 1));
    assert_eq!(i.error, None);
    assert_eq!(std::fs::metadata(path).unwrap().len(), 1048576);
    assert_eq!(
        statuses(&take_events("cj0")),
        [
            JobStatus::Created,
            JobStatus::Running,
            JobStatus::Waiting,
            JobStatus::Pending,
            JobStatus::Concluded,
            JobStatus::Null
        ]
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn create_job_errors() {
    let _s = serial();
    let g = Arc::new(BlockGraph::new());
    let o = |s: &str| -> BlockdevCreateOptions { from_json(s) };

    // Drivers without .bdrv_co_create.
    let e = g.blockdev_create_job("cj1", o(r#"{"driver": "null-co"}"#)).unwrap_err();
    assert_eq!(e.message(), "Driver does not support blockdev-create");

    // Job IDs.
    let file = r#"{"driver": "file", "filename": "/nonexistent-dir/x.img", "size": 0}"#;
    let e = g.blockdev_create_job("1x", o(file)).unwrap_err();
    assert_eq!(e.message(), "Invalid job ID '1x'");

    // A failing creation fails the job, which stays for job-dismiss.
    take_events("cj2");
    g.blockdev_create_job("cj2", o(file)).unwrap();
    let e = g.blockdev_create_job("cj2", o(file)).unwrap_err();
    assert_eq!(e.message(), "Job ID 'cj2' already in use");
    let i = run_to_end(&g, "cj2");
    assert_eq!(
        i.error.as_deref(),
        Some("Could not create '/nonexistent-dir/x.img': No such file or directory")
    );
    assert_eq!((i.current_progress, i.total_progress), (1, 1));
    let st = statuses(&take_events("cj2"));
    assert!(st.contains(&JobStatus::Aborting), "{st:?}");
}

#[test]
fn luks_create_and_amend_jobs() {
    let _s = serial();
    setup();
    let dir = scratch("luks");
    let path = dir.join("a.luks");
    let path = path.to_str().unwrap();
    let g = Arc::new(BlockGraph::new());

    g.blockdev_create_job(
        "lf",
        from_json(&format!(r#"{{"driver": "file", "filename": "{path}", "size": 0}}"#)),
    )
    .unwrap();
    assert_eq!(run_to_end(&g, "lf").error, None);
    g.blockdev_create_job(
        "lc",
        from_json(&format!(
            r#"{{"driver": "luks", "size": 65536, "key-secret": "jobsec0", "iter-time": 10,
                "file": {{"driver": "file", "filename": "{path}"}}}}"#
        )),
    )
    .unwrap();
    assert_eq!(run_to_end(&g, "lc").error, None);

    add(
        &g,
        &format!(
            r#"{{"driver": "luks", "node-name": "l0", "key-secret": "jobsec0",
                "file": {{"driver": "file", "node-name": "f0", "filename": "{path}"}}}}"#
        ),
    );
    let node = g.find_node("l0").unwrap();
    assert_eq!(slots(&node)[..3], [true, false, false]);

    let am = |s: &str| -> BlockdevAmendOptions { from_json(s) };

    // Command errors.
    let e = g
        .x_blockdev_amend_job("am", "nope", am(r#"{"driver": "luks", "state": "active"}"#), None)
        .unwrap_err();
    assert_eq!(e.message(), "Cannot find device='' nor node-name='nope'");
    let e = g
        .x_blockdev_amend_job("am", "f0", am(r#"{"driver": "luks", "state": "active"}"#), None)
        .unwrap_err();
    assert_eq!(e.message(), "x-blockdev-amend doesn't support changing the block driver");

    // Erasing the last key slot needs force: the job fails.
    let erase0 = r#"{"driver": "luks", "state": "inactive", "keyslot": 0}"#;
    g.x_blockdev_amend_job("am0", "l0", am(erase0), None).unwrap();
    let i = run_to_end(&g, "am0");
    assert_eq!(i.type_, JobType::Amend);
    assert_eq!(
        i.error.as_deref(),
        Some(
            "Attempt to erase the only active keyslot 0 which will erase all the data in the image irreversibly - refusing operation"
        )
    );
    assert_eq!(slots(&node)[..3], [true, false, false]);

    // A new key slot.
    let add1 = r#"{"driver": "luks", "state": "active", "new-secret": "jobsec1", "iter-time": 10}"#;
    g.x_blockdev_amend_job("am1", "l0", am(add1), Some(false)).unwrap();
    let i = run_to_end(&g, "am1");
    assert_eq!(i.error, None);
    assert_eq!((i.current_progress, i.total_progress), (1, 1));
    assert_eq!(slots(&node)[..3], [true, true, false]);

    // The image still opens with the new secret.
    drop(node);
    g.blockdev_del("l0").unwrap();
    add(
        &g,
        &format!(
            r#"{{"driver": "luks", "node-name": "l1", "key-secret": "jobsec1",
                "file": {{"driver": "file", "filename": "{path}"}}}}"#
        ),
    );
    g.blockdev_del("l1").unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
}
