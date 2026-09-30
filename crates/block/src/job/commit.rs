// SPDX-License-Identifier: GPL-2.0-or-later

//! `block-commit` from block/commit.c and blockdev.c: copies the data of the backing chain
//! between a top node and a base node into the base, and then drops the nodes above the
//! base from the chain. Committing the active layer is a mirror job ([`super::mirror`]).

use std::any::Any;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use ruvm_base::{Error, ErrorClass, Result};
use ruvm_qapi::types::{BlockCommitArg, BlockErrorAction, BlockdevOnError, JobType};

use super::block_job::{
    BlockJobParams, block_job_add_bdrv, block_job_create, device_name, device_or_node_name,
};
use super::blocker::{BlockOpType, freeze_chain, op_is_blocked, unfreeze_chain};
use super::chain::{
    append_filter, chain_contains, drop_intermediate, find_backing_image, find_overlay,
    internal_open,
};
use super::core::{Job, JobDriver, JobErr};
use super::main_loop::bql_lock;
use super::mirror::commit_active_start;
use super::stream::{job_flags, lookup_node};
use crate::backend::BlockBackend;
use crate::drivers::DriverDef;
use crate::graph::BlockGraph;
use crate::node::{BDRV_BLOCK_ALLOCATED, BDRV_BLOCK_ZERO, Driver, Node, errno};
use crate::perm::{
    BLK_PERM_ALL, BLK_PERM_CONSISTENT_READ, BLK_PERM_RESIZE, BLK_PERM_WRITE,
    BLK_PERM_WRITE_UNCHANGED, PermCtx,
};

/// Size of data buffer for populating the image file. This should be large enough to
/// process multiple clusters in a single call, so that populating contiguous regions of the
/// image is efficient.
const COMMIT_BUFFER_SIZE: u64 = 512 * 1024;

/// `bdrv_commit_top`.
pub(crate) static COMMIT_TOP: DriverDef =
    DriverDef::filter("commit_top", internal_open).with_filtered_backing();

/// The `commit_top` driver: reads go to the backing node, and it takes no permissions on it,
/// so that the job can block consistent reads on the chain below.
pub(crate) struct CommitTop;

impl Driver for CommitTop {
    fn pread(&self, bs: &Node, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        match bs.filter_child() {
            Some(c) => c.node.pread(offset, buf),
            None => Err(errno(crate::node::ENOMEDIUM)),
        }
    }

    fn pwrite(&self, _: &Node, _: u64, _: &[u8]) -> io::Result<()> {
        Err(errno(libc::ENOTSUP))
    }

    fn getlength(&self, bs: &Node) -> io::Result<u64> {
        match bs.filter_child() {
            Some(c) => c.node.getlength(),
            None => Ok(bs.total_sectors().max(0) as u64 * 512),
        }
    }

    fn has_truncate(&self) -> bool {
        false
    }

    fn exact_filename(&self, bs: &Node) -> Option<String> {
        bs.filter_child().map(|c| c.node.meta.lock().unwrap().filename.clone())
    }

    fn child_perm_for(&self, _: &PermCtx<'_>, _: u64, _: u64) -> (u64, u64) {
        (0, BLK_PERM_ALL)
    }
}

/// `CommitBlockJob`.
struct CommitJob {
    commit_top_bs: Arc<Node>,
    top: Mutex<Option<BlockBackend>>,
    base: Mutex<Option<BlockBackend>>,
    base_bs: Arc<Node>,
    base_overlay: Arc<Node>,
    on_error: BlockdevOnError,
    base_read_only: bool,
    chain_frozen: AtomicBool,
    backing_file_str: Option<String>,
    backing_mask_protocol: bool,
}

impl CommitJob {
    fn top_bs(&self) -> Arc<Node> {
        self.top.lock().unwrap().as_ref().and_then(|b| b.root()).expect("the job has its top")
    }

    /// `commit_iteration()`: copies what is allocated above the base at `offset`, returns the
    /// number of bytes done.
    fn iteration(&self, job: &Arc<Job>, offset: u64) -> Result<u64, JobErr> {
        let top_bs = self.top_bs();
        let mut error_in_source = true;
        // Copy if allocated above the base. QEMU passes `true` as the mode here.
        let r = top_bs
            .common_block_status_above(
                Some(&self.base_overlay),
                true,
                1,
                offset,
                COMMIT_BUFFER_SIZE,
            )
            .and_then(|(st, _)| {
                let bytes = st.pnum;
                if st.ret & BDRV_BLOCK_ALLOCATED != 0 {
                    let base = self.base.lock().unwrap();
                    let base = base.as_ref().expect("the job has its base");
                    if st.ret & BDRV_BLOCK_ZERO != 0 {
                        // If the top (sub)clusters are smaller than the base (sub)clusters,
                        // this will not unmap unless the underlying device does some tracking
                        // of these requests. Ideally, we would find the maximal extent of
                        // the zero clusters.
                        base.pwrite_zeroes(offset, bytes, true).inspect_err(|_| {
                            error_in_source = false;
                        })?;
                    } else {
                        let mut buf = vec![0u8; bytes as usize];
                        self.top.lock().unwrap().as_ref().expect("top").pread(offset, &mut buf)?;
                        base.pwrite(offset, &buf).inspect_err(|_| error_in_source = false)?;
                    }
                    // Whether zeroes actually end up on disk depends on the details of the
                    // underlying driver. Therefore, this might rate limit more than is
                    // necessary.
                    job.ratelimit_processed_bytes(bytes);
                }
                Ok(bytes)
            });
        match r {
            Ok(bytes) => {
                // Publish progress
                job.progress_update(bytes);
                Ok(bytes)
            }
            Err(e) => {
                let e = e.raw_os_error().unwrap_or(libc::EIO);
                let action = job.error_action(self.on_error, error_in_source, e);
                if action == BlockErrorAction::Report {
                    return Err(JobErr::errno(e));
                }
                Ok(0)
            }
        }
    }
}

impl JobDriver for CommitJob {
    /// `commit_run()`.
    fn run(&self, job: &Arc<Job>) -> Result<(), JobErr> {
        let len = self.top.lock().unwrap().as_ref().expect("top").getlength()?;
        job.progress_set_remaining(len);
        let base_len = self.base.lock().unwrap().as_ref().expect("base").getlength()?;
        if base_len < len {
            self.base
                .lock()
                .unwrap()
                .as_ref()
                .expect("base")
                .truncate(len)
                .map_err(|e| JobErr::with(libc::EIO, e))?;
        }
        let mut offset = 0;
        while offset < len {
            // Note that even when no rate limit is applied we need to yield with no pending
            // I/O here so that bdrv_drain_all() returns.
            job.ratelimit_sleep();
            if job.is_cancelled() {
                break;
            }
            offset += self.iteration(job, offset)?;
        }
        Ok(())
    }

    /// `commit_prepare()`.
    fn prepare(&self, _job: &Arc<Job>) -> Option<Result<(), JobErr>> {
        unfreeze_chain(&self.commit_top_bs, Some(&self.base_bs));
        self.chain_frozen.store(false, Ordering::SeqCst);
        // Remove base node parent that still uses BLK_PERM_WRITE/RESIZE before the normal
        // backing chain can be restored.
        self.base.lock().unwrap().take();
        // FIXME: bdrv_drop_intermediate treats total failures and partial failures
        // identically. Further work is needed to disambiguate these cases.
        Some(
            drop_intermediate(
                &self.commit_top_bs,
                &self.base_bs,
                self.backing_file_str.as_deref(),
                self.backing_mask_protocol,
            )
            .map_err(JobErr::errno),
        )
    }

    /// `commit_abort()`.
    fn abort(&self, job: &Arc<Job>) {
        if self.chain_frozen.load(Ordering::SeqCst) {
            unfreeze_chain(&self.commit_top_bs, Some(&self.base_bs));
        }
        self.base.lock().unwrap().take();
        // free the blockers on the intermediate nodes so that bdrv_replace_nodes can succeed
        job.bj().remove_all_bdrv();
        // If bdrv_drop_intermediate() failed (or was not invoked), remove the commit filter
        // driver from the backing chain now. Do this as the final step so that the
        // 'consistent read' permission can be granted.
        if let Some(backing) = self.commit_top_bs.filter_child().map(|c| c.node) {
            let _d = backing.drained();
            Node::replace_node(&self.commit_top_bs, &backing).expect("dropping commit_top");
        }
    }

    /// `commit_clean()`.
    fn clean(&self, _job: &Arc<Job>) {
        // restore base open flags here if appropriate (e.g., change the base back to r/o).
        // These reopens do not need to be atomic, since we won't abort even on failure here
        if self.base_read_only {
            let _ = self.base_bs.reopen_set_read_only(true);
        }
        self.top.lock().unwrap().take();
    }

    fn as_any(&self) -> Option<&dyn Any> {
        Some(self)
    }
}

/// What `commit_start()` gets.
pub(crate) struct CommitParams<'a> {
    pub job_id: Option<&'a str>,
    pub bs: Arc<Node>,
    pub base: Arc<Node>,
    pub top: Arc<Node>,
    pub creation_flags: u32,
    pub speed: i64,
    pub on_error: BlockdevOnError,
    pub backing_file_str: Option<String>,
    pub backing_mask_protocol: bool,
    pub filter_node_name: Option<&'a str>,
}

/// `commit_start()`.
pub(crate) fn commit_start(graph: &BlockGraph, p: CommitParams<'_>) -> Result<Arc<Job>> {
    let (bs, base, top) = (&p.bs, &p.base, &p.top);
    assert!(!Arc::ptr_eq(top, bs));
    if Arc::ptr_eq(&top.skip_filters(), &base.skip_filters()) {
        return Err(Error::generic("Invalid files for merge: top and base are the same"));
    }
    let base_size =
        base.getlength().map_err(|e| Error::from_io("Could not inquire base image size", e))?;
    let top_size =
        top.getlength().map_err(|e| Error::from_io("Could not inquire top image size", e))?;
    let mut base_perms = BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE;
    if base_size < top_size {
        base_perms |= BLK_PERM_RESIZE;
    }

    let job = block_job_create(
        graph,
        bs,
        BlockJobParams {
            job_id: p.job_id,
            job_type: JobType::Commit,
            txn: None,
            perm: 0,
            shared: BLK_PERM_ALL,
            speed: p.speed,
            flags: p.creation_flags,
            cb: None,
        },
    )?;

    let mut base_read_only = false;
    let mut commit_top_bs: Option<Arc<Node>> = None;
    let mut chain_frozen = false;
    let mut blks: (Option<BlockBackend>, Option<BlockBackend>) = (None, None);
    let r = (|| -> Result<Arc<Node>> {
        // convert base to r/w, if necessary
        base_read_only = base.read_only();
        if base_read_only {
            base.reopen_set_read_only(false)?;
        }
        // Insert commit_top block node above top, so we can block consistent read on the
        // backing chain below it
        let ct = append_filter(graph, top, p.filter_node_name, &COMMIT_TOP, Box::new(CommitTop))?;
        commit_top_bs = Some(ct.clone());

        // Block all nodes between top and base, because they will disappear from the chain
        // after this operation. Note that this assumes that the user is fine with removing
        // all nodes (including R/W filters) between top and base. Assuring this is the
        // responsibility of the interface (i.e. whoever calls commit_start()).
        let base_overlay = find_overlay(top, Some(base)).expect("base is below top");
        // The topmost node with bdrv_skip_filters(filtered_base) == bdrv_skip_filters(base)
        let filtered_base = base_overlay.cow_child().map(|c| c.node);
        // XXX BLK_PERM_WRITE needs to be allowed so we don't block ourselves at s->base (if
        // writes are blocked for a node, they are also blocked for its backing file). The
        // other options would be a second filter driver above s->base.
        let mut iter_shared_perms = BLK_PERM_WRITE_UNCHANGED | BLK_PERM_WRITE;
        let mut iter = Some(top.clone());
        while let Some(it) = iter {
            if Arc::ptr_eq(&it, base) {
                break;
            }
            if filtered_base.as_ref().is_some_and(|f| Arc::ptr_eq(f, &it)) {
                // From here on, all nodes are filters on the base. This allows us to share
                // BLK_PERM_CONSISTENT_READ.
                iter_shared_perms |= BLK_PERM_CONSISTENT_READ;
            }
            block_job_add_bdrv(&job, "intermediate node", &it, 0, iter_shared_perms)?;
            iter = it.filter_or_cow_bs();
        }
        freeze_chain(&ct, Some(base))?;
        chain_frozen = true;
        block_job_add_bdrv(&job, "base", base, 0, BLK_PERM_ALL)?;

        let b = BlockBackend::with_node(
            None,
            base.clone(),
            base_perms,
            BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE_UNCHANGED,
        )?;
        b.set_disable_request_queuing(true);
        blks.0 = Some(b);
        // Required permissions are already taken with block_job_add_bdrv()
        let t = BlockBackend::with_node(None, top.clone(), 0, BLK_PERM_ALL)?;
        t.set_disable_request_queuing(true);
        blks.1 = Some(t);
        Ok(base_overlay)
    })();

    match r {
        Ok(base_overlay) => {
            job.set_driver(Box::new(CommitJob {
                commit_top_bs: commit_top_bs.expect("made above"),
                top: Mutex::new(blks.1),
                base: Mutex::new(blks.0),
                base_bs: base.clone(),
                base_overlay,
                on_error: p.on_error,
                base_read_only,
                chain_frozen: AtomicBool::new(true),
                backing_file_str: p.backing_file_str,
                backing_mask_protocol: p.backing_mask_protocol,
            }));
            job.start();
            Ok(job)
        }
        Err(e) => {
            if chain_frozen {
                if let Some(ct) = &commit_top_bs {
                    unfreeze_chain(ct, Some(base));
                }
            }
            drop(blks);
            if base_read_only {
                let _ = base.reopen_set_read_only(true);
            }
            job.early_fail();
            // commit_top_bs has to be replaced after deleting the block job, otherwise this
            // would fail because of lack of permissions.
            if let Some(ct) = commit_top_bs {
                let _d = top.drained();
                Node::replace_node(&ct, top).expect("dropping commit_top");
            }
            Err(e)
        }
    }
}

impl BlockGraph {
    /// `qmp_block_commit()`.
    pub fn block_commit(&self, a: &BlockCommitArg) -> Result<()> {
        let _bql = bql_lock();
        let speed = a.speed.unwrap_or(0);
        let on_error = a.on_error.unwrap_or(BlockdevOnError::Report);
        let flags = job_flags(a.auto_finalize, a.auto_dismiss);
        let backing_mask_protocol = a.backing_mask_protocol.unwrap_or(false);
        let device = a.device.as_str();

        // Important Note: libvirt relies on the DeviceNotFound error class in order to probe
        // for live commit feature versions; for this to work, we must make sure to perform
        // the device lookup before any generic errors that may occur in a scenario in which
        // all optional arguments are omitted.
        let bs = match self.root_bs(device) {
            Ok(bs) => bs,
            Err(e) => {
                if self.lookup_bs(device).is_err() {
                    return Err(Error::new(
                        ErrorClass::DeviceNotFound,
                        format!("Device '{device}' not found"),
                    ));
                }
                return Err(e);
            }
        };
        op_is_blocked(&bs, BlockOpType::CommitSource, &device_or_node_name(self, &bs))?;

        // default top_bs is the active layer
        let mut top_bs = Some(bs.clone());
        if a.top_node.is_some() && a.top.is_some() {
            return Err(Error::generic("'top-node' and 'top' are mutually exclusive"));
        } else if let Some(top_node) = &a.top_node {
            let t = lookup_node(self, top_node)?;
            if !chain_contains(&bs, &t) {
                return Err(Error::generic(format!(
                    "'{top_node}' is not in this backing file chain"
                )));
            }
            top_bs = Some(t);
        } else if let Some(top) = &a.top {
            // This strcmp() is just a shortcut, there is no need to refresh @bs's filename.
            // If it mismatches, bdrv_find_backing_image() will do the refresh and may still
            // return @bs.
            if bs.meta.lock().unwrap().filename != *top {
                top_bs = find_backing_image(&bs, top);
            }
        }
        let Some(top_bs) = top_bs else {
            return Err(Error::generic(format!(
                "Top image file {} not found",
                a.top.as_deref().unwrap_or("NULL")
            )));
        };

        let base_bs = if a.base_node.is_some() && a.base.is_some() {
            return Err(Error::generic("'base-node' and 'base' are mutually exclusive"));
        } else if let Some(base_node) = &a.base_node {
            let b = lookup_node(self, base_node)?;
            if !chain_contains(&top_bs, &b) {
                return Err(Error::generic(format!(
                    "'{base_node}' is not in this backing file chain"
                )));
            }
            b
        } else if let Some(base) = &a.base {
            find_backing_image(&top_bs, base).ok_or_else(|| {
                Error::generic(format!("Can't find '{base}' in the backing chain"))
            })?
        } else {
            find_overlay(&top_bs, None)
                .ok_or_else(|| Error::generic("There is no backimg image"))?
        };

        let stop = base_bs.filter_or_cow_bs();
        let mut iter = Some(top_bs.clone());
        while let Some(it) = iter {
            if stop.as_ref().is_some_and(|s| Arc::ptr_eq(s, &it)) {
                break;
            }
            op_is_blocked(&it, BlockOpType::CommitTarget, &device_or_node_name(self, &it))?;
            iter = it.filter_or_cow_bs();
        }

        // Do not allow attempts to commit an image into itself
        if Arc::ptr_eq(&top_bs, &base_bs) {
            return Err(Error::generic("cannot commit an image into itself"));
        }

        // Active commit is required if and only if someone has taken a WRITE permission on
        // the top node. Historically, we have always used active commit for top nodes, so
        // continue that practice lest we possibly break clients that rely on this behavior,
        // e.g. to later attach this node to a writing parent. (Active commit is never
        // really wrong.)
        let (top_perm, _) = top_bs.cumulative_perm();
        let is_active = Arc::ptr_eq(&top_bs.skip_filters(), &bs.skip_filters());
        if top_perm & BLK_PERM_WRITE != 0 || is_active {
            if a.backing_file.is_some() {
                return Err(Error::generic(if is_active {
                    "'backing-file' specified, but 'top' is the active layer"
                } else {
                    "'backing-file' specified, but 'top' has a writer on it"
                }));
            }
            let dev;
            let job_id = match a.job_id.as_deref() {
                Some(id) => Some(id),
                None => {
                    // Emulate here what block_job_create() does, because it is possible that
                    // @bs != @top_bs (the block job should be named after @bs, even if
                    // @top_bs is the actual source)
                    dev = device_name(self, &bs);
                    Some(dev.as_str())
                }
            };
            commit_active_start(
                self,
                job_id,
                &top_bs,
                &base_bs,
                flags,
                speed,
                on_error,
                a.filter_node_name.as_deref(),
            )
            .map(drop)
        } else {
            if let Some(overlay_bs) = find_overlay(&bs, Some(&top_bs)) {
                op_is_blocked(
                    &overlay_bs,
                    BlockOpType::CommitTarget,
                    &device_or_node_name(self, &overlay_bs),
                )?;
            }
            commit_start(
                self,
                CommitParams {
                    job_id: a.job_id.as_deref(),
                    bs,
                    base: base_bs,
                    top: top_bs,
                    creation_flags: flags,
                    speed,
                    on_error,
                    backing_file_str: a.backing_file.clone(),
                    backing_mask_protocol,
                    filter_node_name: a.filter_node_name.as_deref(),
                },
            )
            .map(drop)
        }
    }
}
