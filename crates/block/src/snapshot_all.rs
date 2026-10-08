// SPDX-License-Identifier: GPL-2.0-or-later

//! The snapshot side of `savevm`, `loadvm`, `delvm` and the `snapshot-*` jobs: the
//! `bdrv_all_*()` functions of block/snapshot.c, which act on every node a snapshot of the
//! whole VM covers, the VM state channel of migration/channel-block.c, `info snapshots` from
//! block/monitor/block-hmp-cmds.c and the one-step jobs of migration/savevm.c.
//!
//! Differences from QEMU:
//!
//! - `bdrv_next()` lists the roots of block backends first and then the monitor owned nodes
//!   that no backend uses, as QEMU does. Backends are not kept in a list of their own here, so
//!   the roots come in the order their nodes were made rather than the order of the backends.
//! - The VM state channel writes what it is given in one `bdrv_save_vmstate()` per 4 MiB
//!   instead of one per `writev()` of the `QEMUFile`.
//! - The jobs run their work on the job's own thread, not in a bottom half of the main loop.

use std::any::Any;
use std::io::{self, Read, Write};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use ruvm_base::{Error, Result};
use ruvm_qapi::cutils::size_to_str;
use ruvm_qapi::types::JobType;

use crate::graph::BlockGraph;
use crate::job::core::{JOB_MANUAL_DISMISS, Job, JobDriver, JobErr};
use crate::job::main_loop::bql_lock;
use crate::node::{Node, SnapshotEntry};
use crate::vvfat::localtime::Zone;

/// The most one VM state request carries.
const VMSTATE_CHUNK: usize = 4 << 20;

/// What a new snapshot records besides the disk contents, the fields of `QEMUSnapshotInfo`
/// that `save_snapshot()` fills in.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SnapshotParams {
    /// The tag.
    pub name: String,
    /// When the snapshot was taken, seconds since the epoch.
    pub date_sec: u32,
    /// And the nanoseconds of it, which QEMU only has to the microsecond.
    pub date_nsec: u32,
    /// `QEMU_CLOCK_VIRTUAL` when the snapshot was taken.
    pub vm_clock_nsec: u64,
    /// The instruction count under record/replay.
    pub icount: Option<u64>,
}

impl SnapshotParams {
    /// A snapshot taken now at virtual clock `vm_clock_nsec`, named `name` or, without one,
    /// after the local time as `vm-%Y%m%d%H%M%S`.
    pub fn now(name: Option<&str>, vm_clock_nsec: u64) -> Self {
        let now =
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
        let name = match name {
            Some(n) => n.to_string(),
            None => {
                let t = Zone::local().localtime(now.as_secs() as i64);
                format!(
                    "vm-{:04}{:02}{:02}{:02}{:02}{:02}",
                    t.year + 1900,
                    t.mon + 1,
                    t.mday,
                    t.hour,
                    t.min,
                    t.sec
                )
            }
        };
        SnapshotParams {
            name,
            date_sec: now.as_secs() as u32,
            // g_date_time_get_microsecond() * 1000
            date_nsec: now.subsec_micros() * 1000,
            vm_clock_nsec,
            icount: None,
        }
    }

    fn entry(&self, vm_state_size: u64) -> SnapshotEntry {
        SnapshotEntry {
            id_str: String::new(),
            name: self.name.clone(),
            vm_state_size,
            date_sec: self.date_sec,
            date_nsec: self.date_nsec,
            vm_clock_nsec: self.vm_clock_nsec,
            icount: self.icount,
        }
    }
}

/// The node that holds the VM state of a snapshot, as `bdrv_all_find_vmstate_bs()` picks it.
#[derive(Clone)]
pub struct VmStateNode(Arc<Node>);

impl std::fmt::Debug for VmStateNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("VmStateNode").field(&self.0.name).finish()
    }
}

impl VmStateNode {
    /// The node name.
    pub fn name(&self) -> &str {
        &self.0.name
    }

    /// `qemu_fopen_bdrv()`: a channel over the VM state area, from its start.
    pub fn channel(&self) -> VmStateChannel {
        VmStateChannel { bs: self.0.clone(), offset: 0 }
    }

    /// `bdrv_snapshot_find()`: the size of the VM state of the snapshot `name`, or `None`
    /// when there is no such snapshot or the snapshots cannot be listed.
    pub fn snapshot_vm_state_size(&self, name: &str) -> Option<u64> {
        find_by_name(&self.0, name).map(|sn| sn.vm_state_size)
    }
}

/// `QIOChannelBlock`: reads and writes the VM state area of a node one after the other.
pub struct VmStateChannel {
    bs: Arc<Node>,
    offset: u64,
}

impl std::fmt::Debug for VmStateChannel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VmStateChannel")
            .field("node", &self.bs.name)
            .field("offset", &self.offset)
            .finish()
    }
}

impl VmStateChannel {
    /// `qio_channel_block_close()`: flushes the node.
    pub fn close(self) -> Result<()> {
        self.bs.flush().map_err(|e| Error::from_io("Unable to flush VMState", e))
    }
}

impl Write for VmStateChannel {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = buf.len().min(VMSTATE_CHUNK);
        self.bs.save_vmstate(self.offset, &buf[..n])?;
        self.offset += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Read for VmStateChannel {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = buf.len().min(VMSTATE_CHUNK);
        self.bs.load_vmstate(self.offset, &mut buf[..n])?;
        self.offset += n as u64;
        Ok(n)
    }
}

/// A `bdrv_drain_all_begin()` section, ended when dropped. It has to end on the thread that
/// began it.
pub struct DrainAllSection(PhantomData<*const ()>);

impl std::fmt::Debug for DrainAllSection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DrainAllSection")
    }
}

impl Drop for DrainAllSection {
    fn drop(&mut self) {
        crate::drain::drain_all_end();
    }
}

/// `bdrv_snapshot_find()`: the snapshot called `name`. A node whose snapshots cannot be
/// listed has none.
fn find_by_name(bs: &Node, name: &str) -> Option<SnapshotEntry> {
    match bs.snapshot_list() {
        Some(Ok(list)) => list.into_iter().find(|sn| sn.name == name),
        _ => None,
    }
}

/// `bdrv_has_blk()`.
fn has_blk(bs: &Node) -> bool {
    bs.parents().iter().any(|p| p.ops.as_ref().is_some_and(|o| o.is_backend()))
}

/// `bdrv_all_snapshots_includes_bs()`: nodes a block backend uses, and monitor owned nodes
/// with no parent at all.
fn includes_bs(bs: &Node) -> bool {
    if !bs.is_inserted() || bs.read_only() {
        return false;
    }
    has_blk(bs) || bs.parent_count() == 0
}

/// The errno behind a failed snapshot list, for `info snapshots`.
fn list_errno(e: &Error) -> i32 {
    std::error::Error::source(e)
        .and_then(|s| s.downcast_ref::<io::Error>())
        .and_then(io::Error::raw_os_error)
        .unwrap_or(libc::EIO)
}

/// The nodes a snapshot works on: the given device list, or all of `bdrv_next()`.
struct Devices {
    nodes: Vec<Arc<Node>>,
    /// A device list was given, so every node in it takes part.
    explicit: bool,
}

impl Devices {
    /// Whether `bs` takes part, `devices || bdrv_all_snapshots_includes_bs(bs)`.
    fn includes(&self, bs: &Node) -> bool {
        self.explicit || includes_bs(bs)
    }
}

/// Runs `f` as a job of type `job_type`, which reports 0 of 1 and then 1 of 1 as its
/// progress and fails with `f`'s error.
struct OneShotJob {
    f: Mutex<Option<OneShot>>,
}

type OneShot = Box<dyn FnOnce() -> Result<()> + Send>;

impl JobDriver for OneShotJob {
    fn run(&self, job: &Arc<Job>) -> Result<(), JobErr> {
        job.progress_set_remaining(1);
        let f = self.f.lock().unwrap().take().expect("the job runs once");
        let r = f();
        job.progress_update(1);
        // The snapshot jobs return -1 when they fail.
        r.map_err(|e| JobErr::with(libc::EPERM, e))
    }

    fn as_any(&self) -> Option<&dyn Any> {
        Some(self)
    }
}

impl BlockGraph {
    /// `bdrv_next()`: the roots of the block backends, then the monitor owned nodes that no
    /// backend uses.
    fn bdrv_next_all(&self) -> Vec<Arc<Node>> {
        let owned: std::collections::BTreeSet<String> =
            self.nodes().into_iter().filter(|n| n.monitor_owned).map(|n| n.node_name).collect();
        let all = self.named_nodes();
        let mut out: Vec<Arc<Node>> = all.iter().filter(|n| has_blk(n)).cloned().collect();
        out.extend(all.into_iter().filter(|n| owned.contains(&n.name) && !has_blk(n)));
        out
    }

    /// `bdrv_all_get_snapshot_devices()`.
    fn snapshot_devices(&self, devices: Option<&[String]>) -> Result<Devices> {
        let Some(devices) = devices else {
            return Ok(Devices { nodes: self.bdrv_next_all(), explicit: false });
        };
        if devices.is_empty() {
            return Err(Error::generic("At least one device is required for snapshot"));
        }
        let mut nodes = Vec::with_capacity(devices.len());
        for d in devices {
            match self.find_node(d) {
                Some(bs) => nodes.push(bs),
                None => return Err(Error::generic(format!("No block device node '{d}'"))),
            }
        }
        Ok(Devices { nodes, explicit: true })
    }

    /// `bdrv_get_device_name()`: the name of the named block backend whose root `bs` is, or
    /// an empty string.
    fn device_name(&self, bs: &Arc<Node>) -> String {
        let backends = self.backends.lock().unwrap();
        for (name, blk) in backends.iter() {
            if blk.root().is_some_and(|r| Arc::ptr_eq(&r, bs)) {
                return name.clone();
            }
        }
        String::new()
    }

    /// `bdrv_drain_all_begin()`, until the section is dropped.
    pub fn drain_all_section(&self) -> DrainAllSection {
        crate::drain::drain_all_begin();
        DrainAllSection(PhantomData)
    }

    /// `bdrv_all_can_snapshot()`: whether every node the snapshot covers can take one.
    /// `devices` is the list of node names `snapshot-*` give, `None` for all disks.
    pub fn all_can_snapshot(&self, devices: Option<&[String]>) -> Result<()> {
        let _g = crate::graph_lock::rdlock();
        let d = self.snapshot_devices(devices)?;
        for bs in &d.nodes {
            if d.includes(bs) && !bs.can_snapshot() {
                return Err(Error::generic(format!(
                    "Device '{}' is writable but does not support snapshots",
                    self.device_or_node_name(bs)
                )));
            }
        }
        Ok(())
    }

    /// `bdrv_all_delete_snapshot()`: deletes the snapshot `name` wherever it is.
    pub fn all_delete_snapshot(&self, name: &str, devices: Option<&[String]>) -> Result<()> {
        let _drain = self.drain_all_section();
        let _g = crate::graph_lock::rdlock();
        let d = self.snapshot_devices(devices)?;
        for bs in &d.nodes {
            if !d.includes(bs) {
                continue;
            }
            let Some(sn) = find_by_name(bs, name) else { continue };
            let dev = self.device_or_node_name(bs);
            bs.snapshot_delete(Some(&sn.id_str), Some(&sn.name), &dev).map_err(|e| {
                e.prepend(format!("Could not delete snapshot '{name}' on '{dev}': "))
            })?;
        }
        Ok(())
    }

    /// `bdrv_all_goto_snapshot()`: reverts every node to the snapshot `name`.
    pub fn all_goto_snapshot(&self, name: &str, devices: Option<&[String]>) -> Result<()> {
        let d = {
            let _g = crate::graph_lock::rdlock();
            self.snapshot_devices(devices)?
        };
        for bs in &d.nodes {
            let includes = {
                let _g = crate::graph_lock::rdlock();
                d.includes(bs)
            };
            if !includes {
                continue;
            }
            bs.snapshot_goto(name).map_err(|e| {
                let _g = crate::graph_lock::rdlock();
                e.prepend(format!(
                    "Could not load snapshot '{name}' on '{}': ",
                    self.device_or_node_name(bs)
                ))
            })?;
        }
        Ok(())
    }

    /// `bdrv_all_has_snapshot()`: whether every node the snapshot covers has one called
    /// `name`.
    pub fn all_has_snapshot(&self, name: &str, devices: Option<&[String]>) -> Result<bool> {
        let _g = crate::graph_lock::rdlock();
        let d = self.snapshot_devices(devices)?;
        Ok(d.nodes.iter().all(|bs| !d.includes(bs) || find_by_name(bs, name).is_some()))
    }

    /// `bdrv_all_create_snapshot()`: takes the snapshot `sn` of every node it covers, with
    /// `vm_state_size` bytes of VM state on `vm_state`.
    pub fn all_create_snapshot(
        &self,
        sn: &SnapshotParams,
        vm_state: &VmStateNode,
        vm_state_size: u64,
        devices: Option<&[String]>,
    ) -> Result<()> {
        let _g = crate::graph_lock::rdlock();
        let d = self.snapshot_devices(devices)?;
        for bs in &d.nodes {
            let r = if Arc::ptr_eq(bs, &vm_state.0) {
                bs.snapshot_create(&sn.entry(vm_state_size))
            } else if d.includes(bs) {
                bs.snapshot_create(&sn.entry(0))
            } else {
                continue;
            };
            if !matches!(r, Some(Ok(()))) {
                return Err(Error::generic(format!(
                    "Could not create snapshot '{}' on '{}'",
                    sn.name,
                    self.device_or_node_name(bs)
                )));
            }
        }
        Ok(())
    }

    /// `bdrv_all_find_vmstate_bs()`: the node `vmstate`, or else the first node that can hold
    /// the VM state.
    pub fn all_find_vmstate_bs(
        &self,
        vmstate: Option<&str>,
        devices: Option<&[String]>,
    ) -> Result<VmStateNode> {
        let _g = crate::graph_lock::rdlock();
        let d = self.snapshot_devices(devices)?;
        for bs in &d.nodes {
            let found = d.includes(bs) && bs.can_snapshot();
            match vmstate {
                Some(v) if v == bs.name => {
                    if found {
                        return Ok(VmStateNode(bs.clone()));
                    }
                    return Err(Error::generic(format!(
                        "vmstate block device '{v}' does not support snapshots"
                    )));
                }
                Some(_) => {}
                None if found => return Ok(VmStateNode(bs.clone())),
                None => {}
            }
        }
        Err(Error::generic(match vmstate {
            Some(v) => format!("vmstate block device '{v}' does not exist"),
            None => "no block device can store vmstate for snapshot".to_string(),
        }))
    }

    /// `hmp_info_snapshots()`: what `info snapshots` prints.
    pub fn info_snapshots(&self) -> String {
        let bs = match self.all_find_vmstate_bs(None, None) {
            Ok(bs) => bs.0,
            Err(e) => return format!("Error: {}\n", e.message()),
        };
        let _g = crate::graph_lock::rdlock();
        let sn_tab = match bs.snapshot_list() {
            Some(Ok(l)) => l,
            Some(Err(e)) => return format!("bdrv_snapshot_list: error {}\n", -list_errno(&e)),
            None => return format!("bdrv_snapshot_list: error {}\n", -libc::ENOTSUP),
        };
        // Every node with snapshots, with its name and its snapshots.
        let mut images: Vec<(String, Vec<SnapshotEntry>)> = Vec::new();
        for bs1 in self.bdrv_next_all() {
            if !bs1.can_snapshot() {
                continue;
            }
            if let Some(Ok(l)) = bs1.snapshot_list() {
                if !l.is_empty() {
                    images.push((self.device_name(&bs1), l));
                }
            }
        }
        if images.is_empty() {
            return "There is no snapshot available.\n".to_string();
        }
        drop(_g);
        let mut global = Vec::new();
        for sn in &sn_tab {
            if self.all_has_snapshot(&sn.name, None).unwrap_or(false) {
                global.push(sn.clone());
                for (_, l) in &mut images {
                    l.retain(|e| e.name != sn.name);
                }
            }
        }
        let mut out = String::from("List of snapshots present on all disks:\n");
        if global.is_empty() {
            out.push_str("None\n");
        } else {
            out.push_str(&snapshot_dump(None));
            out.push('\n');
            for mut sn in global {
                // The ID is not guaranteed to be the same on all images, so overwrite it.
                sn.id_str = "--".to_string();
                out.push_str(&snapshot_dump(Some(&sn)));
                out.push('\n');
            }
        }
        for (name, l) in &images {
            if l.is_empty() {
                continue;
            }
            out.push_str(&format!("\nList of partial (non-loadable) snapshots on '{name}':\n"));
            out.push_str(&snapshot_dump(None));
            out.push('\n');
            for sn in l {
                out.push_str(&snapshot_dump(Some(sn)));
                out.push('\n');
            }
        }
        out
    }

    /// Starts the job `job_id` of type `job_type` (one of the `snapshot-*` types) that runs
    /// `f` once. The job waits in CONCLUDED for `job-dismiss`.
    pub fn start_oneshot_job(
        &self,
        job_id: &str,
        job_type: JobType,
        f: Box<dyn FnOnce() -> Result<()> + Send>,
    ) -> Result<()> {
        let _bql = bql_lock();
        let job = Job::create(Some(job_id), job_type, None, JOB_MANUAL_DISMISS, None, None)?;
        job.set_driver(Box::new(OneShotJob { f: Mutex::new(Some(f)) }));
        job.start();
        Ok(())
    }
}

/// `bdrv_snapshot_dump()`: one row, or the header with `None`, without the newline.
fn snapshot_dump(sn: Option<&SnapshotEntry>) -> String {
    let Some(sn) = sn else {
        return format!(
            "{:<7} {:<16} {:>8} {:>19} {:>15} {:>10}",
            "ID", "TAG", "VM_SIZE", "DATE", "VM_CLOCK", "ICOUNT"
        );
    };
    let t = Zone::local().localtime(i64::from(sn.date_sec));
    let date = format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        t.year + 1900,
        t.mon + 1,
        t.mday,
        t.hour,
        t.min,
        t.sec
    );
    let secs = sn.vm_clock_nsec / 1_000_000_000;
    let clock = format!(
        "{:04}:{:02}:{:02}.{:03}",
        (secs / 3600) as i32,
        ((secs / 60) % 60) as i32,
        (secs % 60) as i32,
        ((sn.vm_clock_nsec / 1_000_000) % 1000) as i32
    );
    let icount = match sn.icount {
        Some(i) => i.to_string(),
        None => "--".to_string(),
    };
    format!(
        "{:<7} {:<16} {:>8} {:>19} {:>15} {:>10}",
        sn.id_str,
        sn.name,
        size_to_str(sn.vm_state_size),
        date,
        clock,
        icount
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dump_rows() {
        assert_eq!(
            snapshot_dump(None),
            "ID      TAG               VM_SIZE                DATE        VM_CLOCK     ICOUNT"
        );
        let sn = SnapshotEntry {
            id_str: "1".into(),
            name: "snap".into(),
            vm_state_size: 3 << 20,
            date_sec: 0,
            date_nsec: 0,
            vm_clock_nsec: 3_723_456_000_000,
            icount: None,
        };
        let row = snapshot_dump(Some(&sn));
        assert!(row.starts_with("1       snap                3 MiB "), "{row}");
        assert!(row.ends_with(" 0001:02:03.456         --"), "{row}");
    }

    #[test]
    fn autoname() {
        let p = SnapshotParams::now(None, 5);
        assert!(p.name.starts_with("vm-") && p.name.len() == 17, "{}", p.name);
        assert_eq!(p.vm_clock_nsec, 5);
        assert_eq!(p.date_nsec % 1000, 0);
        assert_eq!(SnapshotParams::now(Some("x"), 0).name, "x");
    }

    #[test]
    fn device_errors() {
        let g = BlockGraph::new();
        let err = |r: Result<()>| r.unwrap_err().message().to_string();
        assert_eq!(
            err(g.all_can_snapshot(Some(&[]))),
            "At least one device is required for snapshot"
        );
        assert_eq!(err(g.all_can_snapshot(Some(&["nope".into()]))), "No block device node 'nope'");
        assert!(g.all_can_snapshot(None).is_ok());
        assert_eq!(
            g.all_find_vmstate_bs(None, None).unwrap_err().message(),
            "no block device can store vmstate for snapshot"
        );
        assert_eq!(
            g.all_find_vmstate_bs(Some("d"), None).unwrap_err().message(),
            "vmstate block device 'd' does not exist"
        );
        assert_eq!(g.info_snapshots(), "Error: no block device can store vmstate for snapshot\n");
        assert!(g.all_has_snapshot("s", None).unwrap());
    }
}
