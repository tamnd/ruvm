// SPDX-License-Identifier: GPL-2.0-or-later

//! `drive-backup` and `blockdev-backup` from block/backup.c and blockdev.c: a job that
//! puts a `copy-before-write` filter over the source, so that guest writes copy the old
//! data to the target first, and meanwhile copies the rest of the source to the target
//! with block-copy. The target ends up holding the source as it was when the job started.
//!
//! `sync=full` copies everything, `sync=top` only what is allocated in the top image,
//! `sync=bitmap` (and `incremental`) what a dirty bitmap marks, and `sync=none` nothing but
//! what the guest overwrites. With a bitmap, `bitmap-mode` says what happens to the bitmap
//! when the job ends.
//!
//! Differences from QEMU:
//!
//! - There are no transactions, so the actions of `transaction` do not exist and each
//!   command creates and starts its job alone.
//! - `backup_do_checkpoint()`, which only COLO uses, is not there.
//! - There is no `bdrv_cancel_in_flight()`: requests are synchronous, so cancelling waits
//!   for the copy in flight.
//! - "Device has no medium" cannot happen: a node here always has its driver.
//! - `x-perf.use-copy-range` is accepted and ignored, see [`crate::block_copy`].

use std::any::Any;
use std::sync::{Arc, Mutex};

use ruvm_base::{Error, Result};
use ruvm_qapi::types::{
    BitmapSyncMode, BlockErrorAction, BlockdevBackup, BlockdevOnError, DriveBackup, JobType,
    MirrorSyncMode, NewImageMode, OnCbwError,
};
use ruvm_qapi::{QDict, QValue};

use super::block_job::{
    BlockJobParams, block_job_add_bdrv, block_job_create, device_name, device_or_node_name,
};
use super::blocker::{BlockOpType, op_is_blocked};
use super::core::{Job, JobDriver, JobErr};
use super::main_loop::bql_lock;
use super::mirror::img_create;
use super::stream::job_flags;
use crate::bitmap::{BDRV_BITMAP_ALLOW_RO, BDRV_BITMAP_DEFAULT, DirtyBitmap};
use crate::block_copy::{BLOCK_COPY_MAX_WORKERS, BlockCopyState, CallState, block_copy_async};
use crate::filter::copy_before_write::{cbw_append, cbw_drop};
use crate::graph::{BlockGraph, OpenCtx};
use crate::node::Node;
use crate::perm::BLK_PERM_ALL;

/// `BackupBlockJob`.
struct BackupJob {
    cbw: Mutex<Option<Arc<Node>>>,
    sync_bitmap: Option<Arc<DirtyBitmap>>,
    sync_mode: MirrorSyncMode,
    bitmap_mode: BitmapSyncMode,
    on_source_error: BlockdevOnError,
    on_target_error: BlockdevOnError,
    len: u64,
    cluster_size: u64,
    /// `perf.max_chunk`.
    max_chunk: u64,
    bcs: Arc<BlockCopyState>,
    /// `bg_bcs_call`.
    bg_call: Mutex<Option<Arc<CallState>>>,
}

impl BackupJob {
    /// `backup_cleanup_sync_bitmap()`.
    fn cleanup_sync_bitmap(&self, ok: bool) {
        let Some(sb) = &self.sync_bitmap else {
            return;
        };
        let sync = (ok || self.bitmap_mode == BitmapSyncMode::Always)
            && self.bitmap_mode != BitmapSyncMode::Never;
        let bm = if sync {
            // We succeeded, or we always intended to sync the bitmap. Delete this bitmap
            // and install the child.
            sb.abdicate()
        } else {
            // We failed, or we never intended to sync the bitmap anyway. Merge the successor
            // back into the parent, keeping all data.
            sb.reclaim().map(|()| sb.clone())
        }
        .expect("the sync bitmap has a successor while the job runs");

        if !ok && self.bitmap_mode == BitmapSyncMode::Always {
            // If we failed and synced, merge in the bits we didn't copy:
            bm.merge_internal(self.bcs.dirty_bitmap(), true);
        }
    }

    /// `backup_error_action()`.
    fn error_action(&self, job: &Arc<Job>, read: bool, error: i32) -> BlockErrorAction {
        if read {
            job.error_action(self.on_source_error, true, error)
        } else {
            job.error_action(self.on_target_error, false, error)
        }
    }

    /// `backup_loop()`.
    fn backup_loop(&self, job: &Arc<Job>) -> Result<(), JobErr> {
        let ret;
        loop {
            // retry loop
            let weak = Arc::downgrade(job);
            let s = block_copy_async(
                &self.bcs,
                0,
                self.len.div_ceil(self.cluster_size) * self.cluster_size,
                self.max_chunk,
                // backup_block_copy_callback()
                Box::new(move || {
                    if let Some(j) = weak.upgrade() {
                        j.enter();
                    }
                }),
            );
            *self.bg_call.lock().unwrap() = Some(s.clone());

            while !s.finished() && !job.is_cancelled() {
                job.yield_unless(|| s.finished());
            }

            if !s.finished() {
                assert!(job.is_cancelled());
                s.cancel();
                s.wait(None);
                ret = Ok(());
                break;
            }

            if job.is_cancelled() || s.succeeded() {
                ret = Ok(());
                break;
            }

            if s.cancelled() {
                // Job is not cancelled but only block-copy call. This is possible after job
                // pause. Now the pause is finished, start new block-copy iteration.
                continue;
            }

            // The only remaining case is failed block-copy call.
            assert!(s.failed());
            let (error, error_is_read) = s.status();
            match self.error_action(job, error_is_read, error) {
                BlockErrorAction::Report => {
                    ret = Err(JobErr::errno(error));
                    break;
                }
                // Go to pause prior to starting new block-copy call on the next iteration.
                BlockErrorAction::Stop => job.pause_point(),
                // Proceed to new block-copy call to retry.
                BlockErrorAction::Ignore => {}
            }
        }
        *self.bg_call.lock().unwrap() = None;
        ret
    }

    /// `backup_init_bcs_bitmap()`.
    fn init_bcs_bitmap(&self, job: &Arc<Job>) {
        let bcs_bitmap = self.bcs.dirty_bitmap();
        if self.sync_mode == MirrorSyncMode::Bitmap {
            let sb = self.sync_bitmap.as_ref().expect("sync=bitmap has a bitmap");
            bcs_bitmap.clear(false);
            bcs_bitmap.merge_internal(sb, true);
        } else if self.sync_mode == MirrorSyncMode::Top {
            // We can't hog the coroutine to initialize this thoroughly. Set a flag and
            // resume work when we are able to yield safely.
            self.bcs.set_skip_unallocated(true);
        }
        job.progress_set_remaining(bcs_bitmap.count());
    }
}

impl JobDriver for BackupJob {
    /// `backup_run()`.
    fn run(&self, job: &Arc<Job>) -> Result<(), JobErr> {
        self.init_bcs_bitmap(job);

        if self.sync_mode == MirrorSyncMode::Top {
            let mut offset = 0;
            while offset < self.len {
                if job.is_cancelled() {
                    return Err(JobErr::errno(libc::ECANCELED));
                }
                job.pause_point();
                if job.is_cancelled() {
                    return Err(JobErr::errno(libc::ECANCELED));
                }
                let (_, count) = self.bcs.reset_unallocated(offset)?;
                offset += count;
            }
            self.bcs.set_skip_unallocated(false);
        }

        if self.sync_mode == MirrorSyncMode::None {
            // All bits are set in bcs bitmap to allow any cluster to be copied. This does
            // not actually require them to be copied.
            while !job.is_cancelled() {
                // Yield until the job is cancelled. We just let our before_write notify
                // callback service CoW requests.
                job.yield_();
            }
            Ok(())
        } else {
            self.backup_loop(job)
        }
    }

    /// `backup_pause()`.
    fn pause(&self, _job: &Arc<Job>) {
        let call = self.bg_call.lock().unwrap().clone();
        if let Some(c) = call {
            if !c.finished() {
                c.cancel();
                c.wait(None);
            }
        }
    }

    /// `backup_commit()`.
    fn commit(&self, _job: &Arc<Job>) {
        self.cleanup_sync_bitmap(true);
    }

    /// `backup_abort()`.
    fn abort(&self, _job: &Arc<Job>) {
        self.cleanup_sync_bitmap(false);
    }

    /// `backup_clean()`.
    fn clean(&self, job: &Arc<Job>) {
        job.bj().remove_all_bdrv();
        if let Some(cbw) = self.cbw.lock().unwrap().take() {
            cbw_drop(&cbw);
        }
    }

    /// `backup_cancel()`.
    fn cancel(&self, _job: &Arc<Job>, _force: bool) -> Option<bool> {
        Some(true)
    }

    /// `backup_set_speed()`.
    fn set_speed(&self, _job: &Arc<Job>, speed: i64) {
        self.bcs.set_speed(speed.max(0) as u64);
        if let Some(c) = self.bg_call.lock().unwrap().as_ref() {
            c.kick();
        }
    }

    fn as_any(&self) -> Option<&dyn Any> {
        Some(self)
    }
}

/// `BackupPerf` with its defaults filled in.
struct Perf {
    use_copy_range: bool,
    max_workers: i64,
    max_chunk: i64,
    min_cluster_size: u64,
}

/// What `backup_job_create()` gets.
struct BackupParams<'a> {
    job_id: Option<&'a str>,
    bs: Arc<Node>,
    target: Arc<Node>,
    speed: i64,
    sync_mode: MirrorSyncMode,
    sync_bitmap: Option<Arc<DirtyBitmap>>,
    bitmap_mode: BitmapSyncMode,
    compress: bool,
    discard_source: bool,
    filter_node_name: Option<&'a str>,
    perf: Perf,
    on_source_error: BlockdevOnError,
    on_target_error: BlockdevOnError,
    on_cbw_error: OnCbwError,
    creation_flags: u32,
}

/// `bdrv_supports_compressed_writes()`.
fn supports_compressed_writes(bs: &Node) -> bool {
    if !bs.driver.can_compress() {
        return false;
    }
    match bs.filter_child() {
        Some(c) => supports_compressed_writes(&c.node),
        None => true,
    }
}

/// `backup_job_create()`.
fn backup_job_create(graph: &BlockGraph, p: BackupParams<'_>) -> Result<Arc<Job>> {
    let (bs, target) = (&p.bs, &p.target);
    // QMP interface protects us from these cases
    assert!(p.sync_mode != MirrorSyncMode::Incremental);
    assert!(p.sync_bitmap.is_some() || p.sync_mode != MirrorSyncMode::Bitmap);

    if Arc::ptr_eq(bs, target) {
        return Err(Error::generic("Source and target cannot be the same"));
    }
    if !bs.is_inserted() {
        return Err(Error::generic(format!("Device is not inserted: {}", device_name(graph, bs))));
    }
    if !target.is_inserted() {
        return Err(Error::generic(format!(
            "Device is not inserted: {}",
            device_name(graph, target)
        )));
    }
    if p.compress && !supports_compressed_writes(target) {
        return Err(Error::generic(format!(
            "Compression is not supported for this drive {}",
            device_name(graph, target)
        )));
    }
    op_is_blocked(bs, BlockOpType::BackupSource, &device_or_node_name(graph, bs))?;
    op_is_blocked(target, BlockOpType::BackupTarget, &device_or_node_name(graph, target))?;

    if p.perf.max_workers < 1 || p.perf.max_workers > i64::from(i32::MAX) {
        return Err(Error::generic(format!("max-workers must be between 1 and {}", i32::MAX)));
    }
    if p.perf.max_chunk < 0 {
        return Err(Error::generic("max-chunk must be zero (which means no limit) or positive"));
    }

    if let Some(sb) = &p.sync_bitmap {
        // If we need to write to this bitmap, check that we can:
        if p.bitmap_mode != BitmapSyncMode::Never {
            sb.check(BDRV_BITMAP_DEFAULT)?;
        }
        // Create a new bitmap, and freeze/disable this one.
        sb.create_successor()?;
    }

    let mut cbw: Option<Arc<Node>> = None;
    let r = (|| -> Result<(Arc<Job>, Arc<BlockCopyState>, u64, u64)> {
        let name = || device_or_node_name(graph, bs);
        let len = bs
            .getlength()
            .map_err(|e| Error::from_io(format!("Unable to get length for '{}'", name()), e))?;
        let target_len = target
            .getlength()
            .map_err(|e| Error::from_io(format!("Unable to get length for '{}'", name()), e))?;
        if target_len != len {
            return Err(Error::generic("Source and target image have different sizes"));
        }

        let (top, bcs) = cbw_append(
            graph,
            bs,
            target,
            p.filter_node_name,
            p.discard_source,
            p.perf.min_cluster_size,
            p.on_cbw_error,
        )?;
        cbw = Some(top.clone());

        let cluster_size = bcs.cluster_size();
        if p.perf.max_chunk != 0 && (p.perf.max_chunk as u64) < cluster_size {
            return Err(Error::generic(format!(
                "Required max-chunk ({}) is less than backup cluster size ({cluster_size})",
                p.perf.max_chunk
            )));
        }

        // job->len is fixed, so we can't allow resize
        let job = block_job_create(
            graph,
            &top,
            BlockJobParams {
                job_id: p.job_id,
                job_type: JobType::Backup,
                txn: None,
                perm: 0,
                shared: BLK_PERM_ALL,
                speed: p.speed,
                flags: p.creation_flags,
                cb: None,
            },
        )?;
        Ok((job, bcs, cluster_size, len))
    })();

    let (job, bcs, cluster_size, len) = match r {
        Ok(v) => v,
        Err(e) => {
            if let Some(sb) = &p.sync_bitmap {
                let _ = sb.reclaim();
            }
            if let Some(c) = &cbw {
                cbw_drop(c);
            }
            return Err(e);
        }
    };

    bcs.set_copy_opts(p.perf.use_copy_range, p.compress);
    bcs.set_progress_meter(&job);
    bcs.set_speed(p.speed.max(0) as u64);

    // Required permissions are taken by copy-before-write filter target
    block_job_add_bdrv(&job, "target", target, 0, BLK_PERM_ALL)
        .expect("a job can always hold the target without permissions");

    job.set_driver(Box::new(BackupJob {
        cbw: Mutex::new(cbw),
        sync_bitmap: p.sync_bitmap,
        sync_mode: p.sync_mode,
        bitmap_mode: p.bitmap_mode,
        on_source_error: p.on_source_error,
        on_target_error: p.on_target_error,
        len,
        cluster_size,
        max_chunk: p.perf.max_chunk as u64,
        bcs,
        bg_call: Mutex::new(None),
    }));
    Ok(job)
}

/// The fields `DriveBackup` and `BlockdevBackup` share, `BackupCommon`.
struct BackupCommon<'a> {
    job_id: Option<&'a str>,
    sync: MirrorSyncMode,
    speed: Option<i64>,
    bitmap: Option<&'a str>,
    bitmap_mode: Option<BitmapSyncMode>,
    compress: Option<bool>,
    on_source_error: Option<BlockdevOnError>,
    on_target_error: Option<BlockdevOnError>,
    auto_finalize: Option<bool>,
    auto_dismiss: Option<bool>,
    filter_node_name: Option<&'a str>,
    discard_source: Option<bool>,
    x_perf: Option<&'a ruvm_qapi::types::BackupPerf>,
    on_cbw_error: Option<OnCbwError>,
}

macro_rules! backup_common {
    ($b:expr) => {
        BackupCommon {
            job_id: $b.job_id.as_deref(),
            sync: $b.sync,
            speed: $b.speed,
            bitmap: $b.bitmap.as_deref(),
            bitmap_mode: $b.bitmap_mode,
            compress: $b.compress,
            on_source_error: $b.on_source_error,
            on_target_error: $b.on_target_error,
            auto_finalize: $b.auto_finalize,
            auto_dismiss: $b.auto_dismiss,
            filter_node_name: $b.filter_node_name.as_deref(),
            discard_source: $b.discard_source,
            x_perf: $b.x_perf.as_ref(),
            on_cbw_error: $b.on_cbw_error,
        }
    };
}

/// `do_backup_common()`.
fn do_backup_common(
    graph: &BlockGraph,
    b: BackupCommon<'_>,
    bs: &Arc<Node>,
    target: &Arc<Node>,
) -> Result<Arc<Job>> {
    let mut perf = Perf {
        use_copy_range: false,
        max_workers: BLOCK_COPY_MAX_WORKERS,
        max_chunk: 0,
        min_cluster_size: 0,
    };
    if let Some(x) = b.x_perf {
        if let Some(v) = x.use_copy_range {
            perf.use_copy_range = v;
        }
        if let Some(v) = x.max_workers {
            perf.max_workers = v;
        }
        if let Some(v) = x.max_chunk {
            perf.max_chunk = v;
        }
        if let Some(v) = x.min_cluster_size {
            perf.min_cluster_size = v;
        }
    }

    let mut sync = b.sync;
    let mut bitmap_mode = b.bitmap_mode;
    if sync == MirrorSyncMode::Bitmap || sync == MirrorSyncMode::Incremental {
        // done before desugaring 'incremental' to print the right message
        if b.bitmap.is_none() {
            return Err(Error::generic(format!(
                "must provide a valid bitmap name for '{}' sync mode",
                sync.as_str()
            )));
        }
    }

    if sync == MirrorSyncMode::Incremental {
        if bitmap_mode.is_some_and(|m| m != BitmapSyncMode::OnSuccess) {
            return Err(Error::generic(format!(
                "Bitmap sync mode must be '{}' when using sync mode '{}'",
                BitmapSyncMode::OnSuccess.as_str(),
                sync.as_str()
            )));
        }
        sync = MirrorSyncMode::Bitmap;
        bitmap_mode = Some(BitmapSyncMode::OnSuccess);
    }

    let mut bmap = None;
    if let Some(name) = b.bitmap {
        let Some(m) = bs.find_dirty_bitmap(name) else {
            return Err(Error::generic(format!("Bitmap '{name}' could not be found")));
        };
        let Some(mode) = bitmap_mode else {
            return Err(Error::generic("Bitmap sync mode must be given when providing a bitmap"));
        };
        m.check(BDRV_BITMAP_ALLOW_RO)?;

        // This does not produce a useful bitmap artifact:
        if sync == MirrorSyncMode::None {
            return Err(Error::generic(format!(
                "sync mode '{}' does not produce meaningful bitmap outputs",
                sync.as_str()
            )));
        }

        // If the bitmap isn't used for input or output, this is useless:
        if mode == BitmapSyncMode::Never && sync != MirrorSyncMode::Bitmap {
            return Err(Error::generic(format!(
                "Bitmap sync mode '{}' has no meaningful effect when combined with sync mode \
                 '{}'",
                mode.as_str(),
                sync.as_str()
            )));
        }
        bmap = Some(m);
    }

    if b.bitmap.is_none() && bitmap_mode.is_some() {
        return Err(Error::generic("Cannot specify bitmap sync mode without a bitmap"));
    }

    backup_job_create(
        graph,
        BackupParams {
            job_id: b.job_id,
            bs: bs.clone(),
            target: target.clone(),
            speed: b.speed.unwrap_or(0),
            sync_mode: sync,
            sync_bitmap: bmap,
            bitmap_mode: bitmap_mode.unwrap_or(BitmapSyncMode::OnSuccess),
            compress: b.compress.unwrap_or(false),
            discard_source: b.discard_source.unwrap_or(false),
            filter_node_name: b.filter_node_name,
            perf,
            on_source_error: b.on_source_error.unwrap_or(BlockdevOnError::Report),
            on_target_error: b.on_target_error.unwrap_or(BlockdevOnError::Report),
            on_cbw_error: b.on_cbw_error.unwrap_or(OnCbwError::BreakGuestWrite),
            creation_flags: job_flags(b.auto_finalize, b.auto_dismiss),
        },
    )
}

impl BlockGraph {
    /// `qmp_drive_backup()`: backs `device` up to a new (or, with `mode=existing`, an
    /// existing) image file.
    pub fn drive_backup(&self, a: &DriveBackup) -> Result<()> {
        let _bql = bql_lock();
        let mode = a.mode.unwrap_or(NewImageMode::AbsolutePaths);
        let bs = self.lookup_bs(&a.device)?;
        let drained = bs.drained();

        let format: Option<String> = match &a.format {
            Some(f) => Some(f.clone()),
            None if mode == NewImageMode::Existing => None,
            None => Some(bs.driver_name.to_string()),
        };

        // Early check to avoid creating target
        op_is_blocked(&bs, BlockOpType::BackupSource, &device_or_node_name(self, &bs))?;

        // See if we have a backing HD we can use to create our new image on top of.
        let mut sync = a.sync;
        let mut source = None;
        let mut no_backing = false;
        if sync == MirrorSyncMode::Top {
            // Backup will not replace the source by the target, so none of the filters
            // skipped here will be removed (in contrast to mirror). Therefore, we can skip
            // all of them when looking for the first COW relationship.
            source = bs.skip_filters().cow_child().map(|c| c.node);
            if source.is_none() {
                sync = MirrorSyncMode::Full;
            }
        }
        if sync == MirrorSyncMode::None {
            source = Some(bs.clone());
            no_backing = true;
        }

        let size = bs.getlength().map_err(|e| Error::from_io("bdrv_getlength failed", e))?;

        if mode != NewImageMode::Existing {
            let fmt = format.as_deref().expect("a format for a new image");
            match &source {
                Some(src) => {
                    // Implicit filters should not appear in the filename
                    src.refresh_filename();
                    let backing_name = src.meta.lock().unwrap().filename.clone();
                    img_create(
                        self,
                        &a.target,
                        fmt,
                        Some(&backing_name),
                        Some(src.driver_name),
                        size,
                    )?;
                }
                None => img_create(self, &a.target, fmt, None, None, size)?,
            }
        }

        let mut options = QDict::new();
        options.put("discard", "unmap");
        options.put("detect-zeroes", "unmap");
        if let Some(f) = &format {
            options.put("driver", f.as_str());
        }
        if no_backing {
            options.put("backing", QValue::Null);
        }
        let (target_bs, _, _) =
            self.open_nodes_qdict(Some(&a.target), options, OpenCtx::default())?;

        if no_backing {
            target_bs.set_backing_hd(source.clone())?;
        }

        let mut common = backup_common!(a);
        common.sync = sync;
        let job = do_backup_common(self, common, &bs, &target_bs)?;
        drop(target_bs);
        drop(drained);
        job.start();
        Ok(())
    }

    /// `qmp_blockdev_backup()`: backs `device` up to the node `target`.
    pub fn blockdev_backup(&self, a: &BlockdevBackup) -> Result<()> {
        let _bql = bql_lock();
        let bs = self.lookup_bs(&a.device)?;
        let target_bs = self.lookup_bs(&a.target)?;
        let drained = bs.drained();
        let job = do_backup_common(self, backup_common!(a), &bs, &target_bs)?;
        drop(drained);
        job.start();
        Ok(())
    }
}
