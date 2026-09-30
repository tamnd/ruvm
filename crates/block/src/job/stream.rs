// SPDX-License-Identifier: GPL-2.0-or-later

//! `block-stream` from block/stream.c and blockdev.c: copies the data of the backing chain
//! between a node and a base into the node, through a `copy-on-read` filter, and then makes
//! the base the backing node of the node.

use std::any::Any;
use std::sync::{Arc, Mutex};

use ruvm_base::{Error, Result};
use ruvm_qapi::QDict;
use ruvm_qapi::types::{BlockErrorAction, BlockStreamArg, BlockdevOnError, JobType};

use super::block_job::{BlockJobParams, block_job_add_bdrv, block_job_create, device_or_node_name};
use super::blocker::{BlockOpType, freeze_chain, op_is_blocked, unfreeze_chain};
use super::chain::{chain_contains, drop_filter, find_backing_image, find_overlay, insert_node};
use super::core::{JOB_DEFAULT, JOB_MANUAL_DISMISS, JOB_MANUAL_FINALIZE, Job, JobDriver, JobErr};
use super::main_loop::bql_lock;
use crate::backend::BlockBackend;
use crate::graph::BlockGraph;
use crate::node::Node;
use crate::perm::{
    BLK_PERM_ALL, BLK_PERM_CONSISTENT_READ, BLK_PERM_WRITE, BLK_PERM_WRITE_UNCHANGED,
};

/// Maximum chunk size to feed to copy-on-read. This should be large enough to process
/// multiple clusters in a single call, so that populating contiguous regions of the image
/// is efficient.
const STREAM_CHUNK: u64 = 512 * 1024;

/// `StreamBlockJob`.
struct StreamJob {
    blk: Mutex<Option<BlockBackend>>,
    /// COW overlay (stream from this)
    base_overlay: Arc<Node>,
    /// Node directly above the base
    above_base: Arc<Node>,
    cor_filter: Mutex<Option<Arc<Node>>>,
    target: Arc<Node>,
    on_error: BlockdevOnError,
    backing_file_str: Option<String>,
    backing_mask_protocol: bool,
    bs_read_only: bool,
}

/// `bdrv_cor_filter_drop()`.
fn cor_filter_drop(f: &Arc<Node>) {
    // A filter can always be dropped, its child takes its parents.
    let _ = drop_filter(f);
}

impl StreamJob {
    /// `stream_populate()`: a copy-on-read read through the filter.
    fn populate(&self, offset: u64, bytes: u64) -> std::io::Result<()> {
        let cor = self.cor_filter.lock().unwrap().clone().expect("the filter is in place");
        let mut buf = vec![0u8; bytes as usize];
        cor.pread(offset, &mut buf)
    }
}

impl JobDriver for StreamJob {
    /// `stream_run()`.
    fn run(&self, job: &Arc<Job>) -> Result<(), JobErr> {
        let unfiltered_bs = self.target.skip_filters();
        if Arc::ptr_eq(&unfiltered_bs, &self.base_overlay) {
            // Nothing to stream
            return Ok(());
        }
        let len = self.target.getlength()?;
        job.progress_set_remaining(len);

        let mut error = 0;
        let mut offset = 0u64;
        let mut n = 0u64;
        while offset < len {
            // Note that even when no rate limit is applied we need to yield with no pending
            // I/O here so that bdrv_drain_all() returns.
            job.ratelimit_sleep();
            if job.is_cancelled() {
                break;
            }

            let mut copy = false;
            let mut ret: std::io::Result<()> = Ok(());
            match unfiltered_bs.is_allocated(offset, STREAM_CHUNK) {
                // Allocated in the top, no need to copy.
                Ok((true, pnum)) => n = pnum,
                Ok((false, pnum)) => {
                    n = pnum;
                    // Copy if allocated in the intermediate images. Limit to the
                    // known-unallocated area [offset, offset+n*BDRV_SECTOR_SIZE).
                    let cow = unfiltered_bs.cow_child().map(|c| c.node);
                    let r = match &cow {
                        Some(c) => c.is_allocated_above(Some(&self.base_overlay), true, offset, n),
                        None => Ok((0, 0)),
                    };
                    match r {
                        Ok((depth, pnum)) => {
                            n = pnum;
                            // Finish early if end of backing file has been reached
                            if depth == 0 && n == 0 {
                                n = len - offset;
                            }
                            copy = depth > 0;
                        }
                        Err(e) => ret = Err(e),
                    }
                }
                Err(e) => ret = Err(e),
            }
            if copy {
                ret = self.populate(offset, n);
            }
            if let Err(e) = ret {
                let errno = e.raw_os_error().unwrap_or(libc::EIO);
                let action = job.error_action(self.on_error, true, errno);
                if action == BlockErrorAction::Stop {
                    n = 0;
                    continue;
                }
                if error == 0 {
                    error = errno;
                }
                if action == BlockErrorAction::Report {
                    break;
                }
            }

            // Publish progress
            job.progress_update(n);
            if copy {
                job.ratelimit_processed_bytes(n);
            }
            offset += n;
        }

        // Do not remove the backing file if an error was there but ignored.
        if error != 0 { Err(JobErr::errno(error)) } else { Ok(()) }
    }

    /// `stream_prepare()`.
    fn prepare(&self, _job: &Arc<Job>) -> Option<Result<(), JobErr>> {
        let unfiltered_bs = self.target.skip_filters();
        let unfiltered_bs_cow = unfiltered_bs.cow_child().map(|c| c.node);

        // We should drop filter at this point, as filter hold the backing chain
        if let Some(f) = self.cor_filter.lock().unwrap().take() {
            cor_filter_drop(&f);
        }

        let Some(_cow) = unfiltered_bs_cow else {
            return Some(Ok(()));
        };
        // bdrv_set_backing_hd() requires that all block nodes are drained.
        let _d = unfiltered_bs.drained();
        let base = self.above_base.filter_or_cow_bs();
        let unfiltered_base = base.as_ref().map(|b| b.skip_filters());

        let mut base_id = None;
        let mut base_fmt = None;
        if let Some(ub) = &unfiltered_base {
            base_id = Some(match &self.backing_file_str {
                Some(s) => s.clone(),
                None => {
                    ub.refresh_filename();
                    ub.meta.lock().unwrap().filename.clone()
                }
            });
            base_fmt = Some(if self.backing_mask_protocol && ub.is_protocol() {
                "raw"
            } else {
                ub.driver_name
            });
        }

        let local_err = unfiltered_bs.set_backing_hd(base);
        // This call will do I/O, so the graph can change again from here on.
        let ret = unfiltered_bs.change_backing_file(base_id.as_deref(), base_fmt, false);
        if let Err(e) = local_err {
            ruvm_base::report::report_error(&e);
            return Some(Err(JobErr::errno(libc::EPERM)));
        }
        Some(ret.map_err(JobErr::from))
    }

    /// `stream_clean()`.
    fn clean(&self, _job: &Arc<Job>) {
        if let Some(f) = self.cor_filter.lock().unwrap().take() {
            cor_filter_drop(&f);
        }
        self.blk.lock().unwrap().take();
        // Reopen the image back in read-only mode if necessary
        if self.bs_read_only {
            // Give up write permissions before making it read-only
            let _ = self.target.reopen_set_read_only(true);
        }
    }

    fn as_any(&self) -> Option<&dyn Any> {
        Some(self)
    }
}

/// What `stream_start()` gets.
pub(crate) struct StreamParams<'a> {
    pub job_id: Option<&'a str>,
    pub bs: Arc<Node>,
    pub base: Option<Arc<Node>>,
    pub backing_file_str: Option<String>,
    pub backing_mask_protocol: bool,
    pub bottom: Option<Arc<Node>>,
    pub creation_flags: u32,
    pub speed: i64,
    pub on_error: BlockdevOnError,
    pub filter_node_name: Option<&'a str>,
}

/// `stream_start()`.
pub(crate) fn stream_start(graph: &BlockGraph, p: StreamParams<'_>) -> Result<Arc<Job>> {
    let basic_flags = BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE_UNCHANGED;
    let bs = &p.bs;
    assert!(!(p.base.is_some() && p.bottom.is_some()));
    assert!(!(p.backing_file_str.is_some() && p.bottom.is_some()));

    let (base_overlay, above_base) = match &p.bottom {
        Some(bottom) => {
            // New simple interface. The code is written in terms of old interface with @base
            // parameter (still, it doesn't freeze link to base, so in this mean old code is
            // correct for new interface). So, for now, just emulate base_overlay and
            // above_base.
            assert!(!bottom.is_filter());
            (bottom.clone(), bottom.clone())
        }
        None => {
            let Some(base_overlay) = find_overlay(bs, p.base.as_ref()) else {
                let base = p.base.as_ref().expect("a chain always has a bottom");
                return Err(Error::generic(format!(
                    "'{}' is not in the backing chain of '{}'",
                    base.name, bs.name
                )));
            };
            // Find the node directly above @base. @base_overlay is a COW overlay, so it must
            // have a bdrv_cow_child(), but it is the immediate overlay of @base, so between
            // the two there can only be filters.
            let mut above_base = base_overlay.clone();
            let is_base = |n: Option<Arc<Node>>| match (&n, &p.base) {
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                (None, None) => true,
                _ => false,
            };
            if !is_base(above_base.cow_child().map(|c| c.node)) {
                above_base = above_base.cow_child().expect("an overlay").node;
                while !is_base(above_base.filter_child().map(|c| c.node)) {
                    above_base = above_base.filter_child().expect("a filter").node;
                }
            }
            (base_overlay, above_base)
        }
    };

    // Make sure that the image is opened in read-write mode
    let bs_read_only = bs.read_only();
    if bs_read_only {
        // Hold the chain during reopen
        freeze_chain(bs, Some(&above_base))?;
        let r = bs.reopen_set_read_only(false);
        // failure, or cor-filter will hold the chain
        unfreeze_chain(bs, Some(&above_base));
        r?;
    }

    let mut opts = QDict::new();
    opts.put("driver", "copy-on-read");
    opts.put("file", bs.name.as_str());
    // Pass the base_overlay node name as 'bottom' to COR driver
    opts.put("bottom", base_overlay.name.as_str());
    if let Some(n) = p.filter_node_name {
        opts.put("node-name", n);
    }

    let mut job: Option<Arc<Job>> = None;
    let mut cor_filter: Option<Arc<Node>> = None;
    let r = (|| -> Result<(Arc<Job>, BlockBackend)> {
        let cor = insert_node(graph, bs, opts)?;
        cor_filter = Some(cor.clone());

        let j = block_job_create(
            graph,
            &cor,
            BlockJobParams {
                job_id: p.job_id,
                job_type: JobType::Stream,
                txn: None,
                perm: 0,
                shared: BLK_PERM_ALL,
                speed: p.speed,
                flags: p.creation_flags,
                cb: None,
            },
        )?;
        job = Some(j.clone());

        let blk = BlockBackend::with_node(
            None,
            cor.clone(),
            BLK_PERM_CONSISTENT_READ,
            basic_flags | BLK_PERM_WRITE,
        )?;
        // Disable request queuing in the BlockBackend to avoid deadlocks on drain: The job
        // reports that it's busy until it reaches a pause point.
        blk.set_disable_request_queuing(true);

        // Prevent concurrent jobs trying to modify the graph structure here, we already have
        // our own plans. Also don't allow resize as the image size is queried only at the
        // job start and then cached.
        block_job_add_bdrv(&j, "active node", bs, 0, basic_flags | BLK_PERM_WRITE)?;

        // Block all intermediate nodes between bs and base, because they will disappear from
        // the chain after this operation. The streaming job reads every block only once,
        // assuming that it doesn't change, so forbid writes and resizes.
        let base = above_base.filter_or_cow_bs();
        let mut iter = bs.filter_or_cow_bs();
        while let Some(it) = iter {
            if base.as_ref().is_some_and(|b| Arc::ptr_eq(b, &it)) {
                break;
            }
            block_job_add_bdrv(&j, "intermediate node", &it, 0, basic_flags)?;
            iter = it.filter_or_cow_bs();
        }
        Ok((j, blk))
    })();

    match r {
        Ok((j, blk)) => {
            j.set_driver(Box::new(StreamJob {
                blk: Mutex::new(Some(blk)),
                base_overlay,
                above_base,
                cor_filter: Mutex::new(cor_filter),
                target: bs.clone(),
                on_error: p.on_error,
                backing_file_str: p.backing_file_str,
                backing_mask_protocol: p.backing_mask_protocol,
                bs_read_only,
            }));
            j.start();
            Ok(j)
        }
        Err(e) => {
            if let Some(j) = job {
                j.early_fail();
            }
            if let Some(f) = cor_filter {
                cor_filter_drop(&f);
            }
            if bs_read_only {
                let _ = bs.reopen_set_read_only(true);
            }
            Err(e)
        }
    }
}

/// `bdrv_lookup_bs(NULL, name)`.
pub(crate) fn lookup_node(graph: &BlockGraph, name: &str) -> Result<Arc<Node>> {
    graph
        .find_node(name)
        .ok_or_else(|| Error::generic(format!("Cannot find device='' nor node-name='{name}'")))
}

/// The job flags of `auto-finalize` and `auto-dismiss`.
pub(crate) fn job_flags(auto_finalize: Option<bool>, auto_dismiss: Option<bool>) -> u32 {
    let mut f = JOB_DEFAULT;
    if auto_finalize == Some(false) {
        f |= JOB_MANUAL_FINALIZE;
    }
    if auto_dismiss == Some(false) {
        f |= JOB_MANUAL_DISMISS;
    }
    f
}

impl BlockGraph {
    /// `qmp_block_stream()`.
    pub fn block_stream(&self, a: &BlockStreamArg) -> Result<()> {
        let _bql = bql_lock();
        if a.base.is_some() && a.base_node.is_some() {
            return Err(Error::generic(
                "'base' and 'base-node' cannot be specified at the same time",
            ));
        }
        if a.base.is_some() && a.bottom.is_some() {
            return Err(Error::generic("'base' and 'bottom' cannot be specified at the same time"));
        }
        if a.bottom.is_some() && a.base_node.is_some() {
            return Err(Error::generic(
                "'bottom' and 'base-node' cannot be specified at the same time",
            ));
        }
        let device = &a.device;
        let bs = self.lookup_bs(device)?;

        let mut base_bs = None;
        if let Some(base) = &a.base {
            base_bs = Some(find_backing_image(&bs, base).ok_or_else(|| {
                Error::generic(format!("Can't find '{base}' in the backing chain"))
            })?);
        }
        if let Some(base_node) = &a.base_node {
            let b = lookup_node(self, base_node)?;
            if Arc::ptr_eq(&bs, &b) || !chain_contains(&bs, &b) {
                return Err(Error::generic(format!(
                    "Node '{base_node}' is not a backing image of '{device}'"
                )));
            }
            b.refresh_filename();
            base_bs = Some(b);
        }
        let mut bottom_bs = None;
        if let Some(bottom) = &a.bottom {
            let b = lookup_node(self, bottom)?;
            if b.is_filter() {
                return Err(Error::generic(format!(
                    "Node '{bottom}' is a filter, use a non-filter node as 'bottom'"
                )));
            }
            if !chain_contains(&bs, &b) {
                return Err(Error::generic(format!(
                    "Node '{bottom}' is not in a chain starting from '{device}'"
                )));
            }
            bottom_bs = Some(b);
        }

        // Check for op blockers in the whole chain between bs and base (or bottom)
        let iter_end = match &bottom_bs {
            Some(b) => b.filter_or_cow_bs(),
            None => base_bs.clone(),
        };
        let mut iter = Some(bs.clone());
        while let Some(it) = iter {
            if iter_end.as_ref().is_some_and(|e| Arc::ptr_eq(e, &it)) {
                break;
            }
            op_is_blocked(&it, BlockOpType::Stream, &device_or_node_name(self, &it))?;
            iter = it.filter_or_cow_bs();
        }

        // if we are streaming the entire chain, the result will have no backing file, and
        // specifying one is therefore an error
        if base_bs.is_none() && a.backing_file.is_some() {
            return Err(Error::generic("backing file specified, but streaming the entire chain"));
        }

        stream_start(
            self,
            StreamParams {
                job_id: a.job_id.as_deref(),
                bs,
                base: base_bs,
                backing_file_str: a.backing_file.clone(),
                backing_mask_protocol: a.backing_mask_protocol.unwrap_or(false),
                bottom: bottom_bs,
                creation_flags: job_flags(a.auto_finalize, a.auto_dismiss),
                speed: a.speed.unwrap_or(0),
                on_error: a.on_error.unwrap_or(BlockdevOnError::Report),
                filter_node_name: a.filter_node_name.as_deref(),
            },
        )
        .map(drop)
    }
}
