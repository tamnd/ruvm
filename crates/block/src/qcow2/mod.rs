// SPDX-License-Identifier: GPL-2.0-or-later

//! The `qcow2` format driver from block/qcow2.c, block/qcow2-cluster.c, block/qcow2-refcount.c,
//! block/qcow2-snapshot.c, block/qcow2-cache.c and block/qcow2-bitmap.c.
//!
//! The format itself lives in the submodules, which work on a [`State`] and know nothing of
//! the block graph: the image file, the external data file and the backing image are reached
//! through the small [`io::Storage`] and [`io::Backing`] traits. This file is the glue that
//! makes a graph driver out of it, the `BlockDriver bdrv_qcow2` table of QEMU: it turns the
//! typed `blockdev` options into [`OpenOptions`], opens the children, and maps every driver
//! callback onto the state. Encryption goes through [`CryptoProvider`], which uses
//! ruvm-crypto for both the legacy AES scheme and LUKS.
//!
//! Persistent dirty bitmaps are loaded into the state on open and handed over to the node's
//! generic dirty bitmaps (the same `BdrvDirtyBitmap` objects QEMU uses), which then record
//! every write. Whenever the format code needs them, to store them on close or inactivation,
//! to check whether a resize is allowed, or to find room for a new one, the node's persistent
//! bitmaps are copied back into the state for the duration of the call.
//!
//! Differences from QEMU:
//!
//! - QEMU drops `s->lock` around the data transfer of a request so that requests to different
//!   clusters run in parallel. Here one lock serialises all requests of an image.
//! - The loaded bitmaps are given to the node when its limits are first refreshed, right
//!   after the driver's open returns, rather than inside `qcow2_open()`. Nothing can use the
//!   node in between.
//! - `cache-clean-interval` has no timer: a request that finds the interval has passed
//!   cleans the caches (see [`State::maybe_clean_caches`]).
//! - Corruption is reported on stderr only; there is no `BLOCK_IMAGE_CORRUPTED` event yet.
//! - There is no zstd support, as in a QEMU built without it: `compression_type=zstd` is not
//!   even a valid value, and images that use it cannot be opened.
//! - `qcow2_join_options()` is not needed: a reopen gets the full option set from the generic
//!   code already, with the unchanged options filled in.
//! - `qemu-img amend` reports no progress.
//! - The driver keeps the image file and data file as they were when it opened them; a
//!   reopen that replaces `file` updates the state once the reopen is committed.

mod amend;
mod bitmap;
mod cache;
mod check;
mod cluster;
mod compress;
mod create;
mod crypto;
mod header;
mod io;
mod open;
mod refcount;
mod resize;
mod rw;
mod snapshot;
mod state;

use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, MutexGuard};

use ruvm_base::error::strerror;
use ruvm_base::report::error_report;
use ruvm_base::{Error, Result};
use ruvm_qapi::types::{
    BlockMeasureInfo, BlockdevAmendOptions, BlockdevAmendOptionsU, BlockdevCreateOptionsQcow2,
    BlockdevCreateOptionsU, BlockdevOptionsQcow2, BlockdevOptionsU, BlockdevQcow2EncryptionU,
    BlockdevQcow2Version, BlockdevRef, ImageInfoSpecific, ImageInfoSpecificQCow2,
    ImageInfoSpecificQCow2Encryption, ImageInfoSpecificQCow2EncryptionU,
    ImageInfoSpecificQCow2Wrapper, ImageInfoSpecificU, PreallocMode, QCryptoBlockAmendOptionsU,
    QCryptoBlockCreateOptions, QCryptoBlockCreateOptionsU, Qcow2CompressionType,
    Qcow2OverlapChecks,
};
use ruvm_qapi::{QDict, QValue};

use self::bitmap::{Bitmap, persistent_bitmaps_size};
use self::create::{CreateOptions, MeasureSource, validate_cluster_size};
use self::crypto::{CryptoProvider, EncryptionFormat, EncryptionOptions};
use self::io::{Backing, NodeBacking, NodeStorage, Storage};
use self::open::{OVERLAP_OPTION_NAMES, OpenOptions, Prepared};
use self::snapshot::SnapshotInfo;
use self::state::{
    DEFAULT_CLUSTER_SIZE, OpenFlags, QCOW_CRYPT_LUKS, QCOW_CRYPT_NONE, QCOW2_COMPAT_LAZY_REFCOUNTS,
    QCOW2_INCOMPAT_CORRUPT, QCOW2_INCOMPAT_DIRTY, State,
};
use crate::bitmap::PersistentBitmaps;
use crate::drivers::{DriverDef, OpenArgs};
use crate::graph::BlockGraph;
use crate::imgopts::{create_opts_open, take_bool, take_size, take_str, visit_create_options};
use crate::node::{
    BDRV_BLOCK_OFFSET_VALID, BDRV_CHILD_DATA, BDRV_CHILD_IMAGE, BDRV_CHILD_PRIMARY,
    BDRV_REQ_MAY_UNMAP, BDRV_REQ_NO_FALLBACK, BDRV_REQ_ZERO_WRITE, BDRV_SECTOR_SIZE,
    BlockDriverInfo, BlockLimits, BlockStatus, CheckResult, Driver, Node, NodeFlags, ReopenState,
    SnapshotEntry,
};

/// `bdrv_qcow2`.
pub(crate) static QCOW2: DriverDef = DriverDef::format("qcow2", qcow2_open)
    .with_probe(qcow2_probe)
    .with_create_opts(qcow2_co_create_opts)
    .with_create_opts_list(&create::CREATE_OPTS)
    .with_create(qcow2_co_create)
    .with_amend_opts_list(&create::AMEND_OPTS)
    .with_amend_opts(qcow2_amend_options)
    .with_measure(qcow2_measure)
    .with_backing()
    .with_strong_opts(&["encrypt.key-secret"])
    .with_mutable_opts(&MUTABLE_OPTS);

/// `qcow2_mutable_opts`: what a reopen may reset by leaving it out.
static MUTABLE_OPTS: [&str; 21] = [
    "lazy-refcounts",
    "pass-discard-request",
    "pass-discard-snapshot",
    "pass-discard-other",
    "discard-no-unref",
    "overlap-check",
    "overlap-check.template",
    "overlap-check.main-header",
    "overlap-check.active-l1",
    "overlap-check.active-l2",
    "overlap-check.refcount-table",
    "overlap-check.refcount-block",
    "overlap-check.snapshot-table",
    "overlap-check.inactive-l1",
    "overlap-check.inactive-l2",
    "overlap-check.bitmap-directory",
    "cache-size",
    "l2-cache-size",
    "l2-cache-entry-size",
    "refcount-cache-size",
    "cache-clean-interval",
];

/// An open qcow2 node.
struct Qcow2 {
    s: Mutex<State>,
    /// Bitmaps loaded by the open that the node does not have yet.
    pending: Mutex<Vec<Bitmap>>,
}

/// What `reopen_prepare` leaves for `reopen_commit`.
struct Reopen {
    prepared: Prepared,
    options: OpenOptions,
    flags: OpenFlags,
}

/// The locked state. The backing image is attached for as long as the lock is held, because
/// the generic code may change the backing child between requests.
struct Locked<'a>(MutexGuard<'a, State>);

impl Deref for Locked<'_> {
    type Target = State;
    fn deref(&self) -> &State {
        &self.0
    }
}

impl DerefMut for Locked<'_> {
    fn deref_mut(&mut self) -> &mut State {
        &mut self.0
    }
}

impl Drop for Locked<'_> {
    fn drop(&mut self) {
        self.0.backing = None;
    }
}

fn open_flags(f: &NodeFlags) -> OpenFlags {
    OpenFlags {
        read_write: !f.read_only,
        check: f.check,
        no_io: f.no_io,
        inactive: f.inactive,
        unmap: f.unmap,
    }
}

fn on_off(b: bool) -> &'static str {
    if b { "on" } else { "off" }
}

/// An option value as the text `-o` or `-drive` would give it.
fn value_str(v: &QValue) -> String {
    match v {
        QValue::Str(s) => s.clone(),
        QValue::Bool(b) => on_off(*b).to_string(),
        QValue::Int(i) => i.to_string(),
        QValue::Uint(u) => u.to_string(),
        other => other.to_json(),
    }
}

/// The error of a request that only returns `-errno`.
fn errno_err(e: std::io::Error) -> Error {
    Error::with_cause(strerror(&e), e)
}

impl Qcow2 {
    fn lock(&self, bs: &Node) -> Locked<'_> {
        let mut g = self.s.lock().unwrap_or_else(|e| e.into_inner());
        g.backing = bs.backing().map(|c| Arc::new(NodeBacking(c.node)) as Arc<dyn Backing>);
        Locked(g)
    }

    /// The state without the backing image, for callers that have no node.
    fn state(&self) -> MutexGuard<'_, State> {
        self.s.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Hands the bitmaps the open loaded over to the node. Must not be called with the lock
    /// held.
    fn publish_bitmaps(&self, bs: &Node) {
        let pending = std::mem::take(&mut *self.pending.lock().unwrap_or_else(|e| e.into_inner()));
        for b in pending {
            let bm = match bs.create_dirty_bitmap(b.granularity, Some(&b.name)) {
                Ok(bm) => bm,
                Err(e) => {
                    error_report(e.message());
                    continue;
                }
            };
            if !b.inconsistent {
                let mut bits = b.bits;
                bits.resize(bm.serialization_size(0, bm.size()) as usize, 0);
                bm.deserialize_part(&bits, 0, bm.size(), true);
            }
            bm.set_persistence(true);
            if b.inconsistent {
                bm.set_inconsistent();
            }
            if !b.enabled {
                bm.disable();
            }
            if b.readonly {
                bm.set_readonly(true);
            }
        }
    }

    /// The persistent bitmaps of the node, and the loaded ones it does not have yet, in the
    /// form the format code works with.
    fn all_bitmaps(&self, bs: &Node, with_bits: bool) -> Vec<Bitmap> {
        let mut v = node_bitmaps(bs, with_bits);
        v.extend(self.pending.lock().unwrap_or_else(|e| e.into_inner()).iter().cloned());
        v
    }

    /// Runs `f` with the node's persistent bitmaps copied into the state.
    fn with_bitmaps<T>(&self, bs: &Node, with_bits: bool, f: impl FnOnce(&mut State) -> T) -> T {
        let bitmaps = self.all_bitmaps(bs, with_bits);
        let mut s = self.lock(bs);
        s.bitmaps = bitmaps;
        let r = f(&mut s);
        s.bitmaps.clear();
        r
    }

    /// What follows a size change the format code made on its own (snapshot load, amend):
    /// the generic code learns the new length.
    fn size_changed(&self, bs: &Node, old: u64) {
        let new = self.state().total_size;
        if new == old {
            return;
        }
        let _ = bs.refresh_total_sectors(None);
        bs.dirty_bitmap_truncate(new.next_multiple_of(BDRV_SECTOR_SIZE));
        bs.parent_cb_resize();
    }

    /// `qcow2_inactivate()` with the node's bitmaps. Returns whether the bitmaps were stored
    /// and may be released.
    fn do_inactivate(&self, bs: &Node) -> std::io::Result<()> {
        let bitmaps = self.all_bitmaps(bs, true);
        self.pending.lock().unwrap_or_else(|e| e.into_inner()).clear();
        let mut s = self.lock(bs);
        s.bitmaps = bitmaps;
        let r = s.inactivate(&bs.name);
        s.bitmaps.clear();
        r
    }
}

/// The named persistent bitmaps of `bs`. Without `with_bits` the bits are left zero, which is
/// enough for the checks that only look at names, sizes and flags.
fn node_bitmaps(bs: &Node, with_bits: bool) -> Vec<Bitmap> {
    let mut v = Vec::new();
    for bm in bs.dirty_bitmaps() {
        if !bm.get_persistence() {
            continue;
        }
        let Some(name) = bm.name() else { continue };
        let size = bm.size();
        let mut b = Bitmap::new(&name, bm.granularity(), size);
        b.enabled = bm.enabled();
        b.inconsistent = bm.inconsistent();
        b.readonly = bm.readonly();
        if with_bits && !b.inconsistent {
            let len = b.bits.len();
            let mut bits = vec![0u8; bm.serialization_size(0, size) as usize];
            bm.serialize_part(&mut bits, 0, size);
            bits.resize(len, 0);
            b.bits = bits;
        }
        v.push(b);
    }
    v
}

/// `qcow2_probe()`.
fn qcow2_probe(buf: &[u8], _filename: Option<&str>) -> i32 {
    header::probe(buf)
}

/// The typed runtime options as the text options [`OpenOptions::set`] takes.
fn typed_open_options(o: &BlockdevOptionsQcow2, oo: &mut OpenOptions) -> Result<()> {
    let mut pairs: Vec<(String, String)> = Vec::new();
    let mut push = |k: &str, v: String| pairs.push((k.to_string(), v));
    let bools = [
        ("lazy-refcounts", o.lazy_refcounts),
        ("pass-discard-request", o.pass_discard_request),
        ("pass-discard-snapshot", o.pass_discard_snapshot),
        ("pass-discard-other", o.pass_discard_other),
        ("discard-no-unref", o.discard_no_unref),
    ];
    for (k, v) in bools {
        if let Some(v) = v {
            push(k, on_off(v).to_string());
        }
    }
    match &o.overlap_check {
        None => {}
        Some(Qcow2OverlapChecks::Mode(m)) => push("overlap-check", m.as_str().to_string()),
        Some(Qcow2OverlapChecks::Flags(f)) => {
            if let Some(t) = f.template {
                push("overlap-check.template", t.as_str().to_string());
            }
            let flags = [
                f.main_header,
                f.active_l1,
                f.active_l2,
                f.refcount_table,
                f.refcount_block,
                f.snapshot_table,
                f.inactive_l1,
                f.inactive_l2,
                f.bitmap_directory,
            ];
            for (name, v) in OVERLAP_OPTION_NAMES.iter().zip(flags) {
                if let Some(v) = v {
                    push(name, on_off(v).to_string());
                }
            }
        }
    }
    let numbers = [
        ("cache-size", o.cache_size),
        ("l2-cache-size", o.l2_cache_size),
        ("l2-cache-entry-size", o.l2_cache_entry_size),
        ("refcount-cache-size", o.refcount_cache_size),
        ("cache-clean-interval", o.cache_clean_interval),
    ];
    for (k, v) in numbers {
        if let Some(v) = v {
            push(k, v.to_string());
        }
    }
    if let Some(e) = &o.encrypt {
        push("encrypt.format", e.u.tag().as_str().to_string());
        let key_secret = match &e.u {
            BlockdevQcow2EncryptionU::Aes(q) => q.key_secret.clone(),
            BlockdevQcow2EncryptionU::Luks(l) => l.key_secret.clone(),
        };
        if let Some(k) = key_secret {
            push("encrypt.key-secret", k);
        }
    }
    for (k, v) in &pairs {
        oo.set(k, v)?;
    }
    Ok(())
}

/// `qcow2_open()`.
fn qcow2_open(args: &mut OpenArgs<'_>, opts: BlockdevOptionsU) -> Result<Box<dyn Driver>> {
    let BlockdevOptionsU::Qcow2(o) = opts else { unreachable!("qcow2 driver with other options") };
    let o = *o;
    let file = args.open_child((*o.file).clone(), "file", BDRV_CHILD_IMAGE | BDRV_CHILD_PRIMARY)?;
    args.set_backing_option(o.backing.clone().map(|b| *b));

    let flags = open_flags(&args.flags);
    let mut oo = OpenOptions { flags, ..OpenOptions::default() };
    typed_open_options(&o, &mut oo)?;
    oo.encryption = Some(Arc::new(CryptoProvider));

    // Under BDRV_O_NO_IO the data file is not opened at all, so that `qemu-img info` can
    // look at an untrusted image without touching the files it names.
    let mut explicit_data = false;
    if !flags.no_io {
        if let Some(r) = o.data_file {
            let node = args.open_child(*r, "data-file", BDRV_CHILD_DATA)?;
            oo.data_file = Some(Arc::new(NodeStorage(node)));
            explicit_data = true;
        }
    }

    let mut s = State::blank(Arc::new(NodeStorage(file)), flags);
    let graph = args.graph;
    let ctx = args.child_ctx(BDRV_CHILD_DATA);
    let mut opened: Option<Arc<Node>> = None;
    {
        let pending = &mut *args.pending;
        let mut open_data_file = |name: &str| -> Result<Arc<dyn Storage>> {
            let node = graph.open_qdict(Some(name), QDict::new(), ctx, pending, true)?;
            opened = Some(node.clone());
            Ok(Arc::new(NodeStorage(node)))
        };
        s.do_open(&oo, &mut open_data_file)?;
    }
    if let Some(node) = opened {
        args.add_child("data-file", node, BDRV_CHILD_DATA);
    }
    if s.data_file.is_some() && (explicit_data || s.has_data_file()) {
        // No data here.
        for c in args.children.iter_mut().filter(|c| c.0 == "file") {
            c.2 &= !BDRV_CHILD_DATA;
        }
    }

    args.meta.encrypted = s.crypt_method_header != QCOW_CRYPT_NONE;
    if !s.backing_file.is_empty() {
        let format = (!s.backing_format.is_empty()).then(|| s.backing_format.clone());
        let file = s.backing_file.clone();
        args.set_backing_file(&file, format.as_deref());
    }

    let pending = std::mem::take(&mut s.bitmaps);
    Ok(Box::new(Qcow2 { s: Mutex::new(s), pending: Mutex::new(pending) }))
}

impl Driver for Qcow2 {
    fn pread(&self, bs: &Node, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
        self.lock(bs).preadv(offset, buf)
    }

    fn pwrite(&self, bs: &Node, offset: u64, buf: &[u8]) -> std::io::Result<()> {
        self.lock(bs).pwritev(offset, buf)
    }

    fn pwrite_zeroes(
        &self,
        bs: &Node,
        offset: u64,
        bytes: u64,
        may_unmap: bool,
    ) -> std::io::Result<()> {
        self.lock(bs).pwrite_zeroes(offset, bytes, may_unmap)
    }

    fn pdiscard(&self, bs: &Node, offset: u64, bytes: u64) -> std::io::Result<()> {
        self.lock(bs).pdiscard(offset, bytes)
    }

    fn getlength(&self, _bs: &Node) -> std::io::Result<u64> {
        Ok(self.state().total_size)
    }

    fn truncate(&self, bs: &Node, len: u64) -> Result<()> {
        self.truncate_full(bs, len, false, PreallocMode::Off, 0)
    }

    fn supported_zero_flags(&self) -> u32 {
        if self.state().qcow_version >= 3 { BDRV_REQ_MAY_UNMAP | BDRV_REQ_NO_FALLBACK } else { 0 }
    }

    fn has_pwrite_zeroes(&self) -> bool {
        true
    }

    fn pwrite_compressed(&self, bs: &Node, offset: u64, buf: &[u8]) -> Option<std::io::Result<()>> {
        Some(self.lock(bs).pwrite_compressed(offset, buf))
    }

    fn can_compress(&self) -> bool {
        true
    }

    /// `qcow2_co_block_status()`.
    fn block_status(
        &self,
        bs: &Node,
        _want: u32,
        offset: u64,
        bytes: u64,
    ) -> Option<std::io::Result<BlockStatus>> {
        let r = self.lock(bs).block_status(offset, bytes);
        Some(r.map(|st| {
            let file = if st.status & BDRV_BLOCK_OFFSET_VALID != 0 {
                Some(bs.child("data-file").map_or_else(|| bs.file(), |c| c.node))
            } else {
                None
            };
            BlockStatus { ret: st.status, pnum: st.pnum, map: st.map, file }
        }))
    }

    /// `qcow2_co_flush_to_os()`.
    fn flush_to_os(&self, bs: &Node) -> std::io::Result<()> {
        self.lock(bs).write_caches()
    }

    /// `qcow2_refresh_limits()`. This is also where the node gets the bitmaps of the open.
    fn refresh_limits(&self, bs: &Node, bl: &mut BlockLimits) -> Result<()> {
        {
            let s = self.state();
            if let Some(c) = &s.crypto {
                bl.request_alignment = c.sector_size() as u32;
            }
            bl.pwrite_zeroes_alignment = s.subcluster_size as u32;
            bl.pdiscard_alignment = s.cluster_size as u32;
        }
        self.publish_bitmaps(bs);
        Ok(())
    }

    /// `qcow2_close()`.
    fn close(&self, bs: &Node) {
        {
            let s = self.state();
            // A fatal corruption took the driver away (`bs->drv = NULL`) and an inactive
            // image was stored already.
            if s.drv_gone || s.flags.inactive {
                return;
            }
        }
        let _ = self.do_inactivate(bs);
    }

    /// `qcow2_reopen_prepare()`.
    fn reopen_prepare(&self, bs: &Node, state: &mut ReopenState) -> Option<Result<()>> {
        Some(self.reopen_prepare_impl(bs, state))
    }

    /// `qcow2_reopen_commit()`.
    fn reopen_commit(&self, bs: &Node, state: &mut ReopenState) {
        let Some(r) = state.opaque.take().and_then(|o| o.downcast::<Reopen>().ok()) else {
            return;
        };
        let Reopen { prepared, options, flags } = *r;
        let mut s = self.lock(bs);
        s.update_options_commit(prepared);
        s.options = options;
        s.flags.read_write = flags.read_write;
        s.flags.unmap = flags.unmap;
    }

    /// `qcow2_reopen_commit_post()`.
    fn reopen_commit_post(&self, bs: &Node) {
        let file = bs.file();
        let file_writable = !file.read_only();
        // A reopen may have replaced the `file` child.
        self.lock(bs).file = Arc::new(NodeStorage(file));
        if bs.read_only() {
            return;
        }
        let filename = bs.filename().unwrap_or_default();
        let find =
            |name: &str| bs.find_dirty_bitmap(name).map(|b| (b.readonly(), b.inconsistent()));
        let r = self.lock(bs).reopen_bitmaps_rw(&filename, file_writable, &find);
        match r {
            Ok(names) => {
                for n in names {
                    if let Some(b) = bs.find_dirty_bitmap(&n) {
                        b.set_readonly(false);
                    }
                }
            }
            Err(e) => error_report(&format!(
                "{}: Failed to make dirty bitmaps writable: {}",
                bs.name,
                e.message()
            )),
        }
    }

    /// `qcow2_reopen_abort()`.
    fn reopen_abort(&self, _bs: &Node, state: &mut ReopenState) {
        state.opaque = None;
    }

    /// `qcow2_get_info()`.
    fn get_info(&self, _bs: &Node) -> Option<std::io::Result<BlockDriverInfo>> {
        let s = self.state();
        Some(Ok(BlockDriverInfo {
            cluster_size: s.cluster_size,
            subcluster_size: s.subcluster_size,
            vm_state_offset: s.vm_state_offset() as i64,
            is_dirty: s.incompatible_features & QCOW2_INCOMPAT_DIRTY != 0,
            needs_compressed_writes: false,
        }))
    }

    fn persistent_bitmaps(&self) -> Option<&dyn PersistentBitmaps> {
        Some(self)
    }

    /// `qcow2_get_specific_info()`.
    fn get_specific_info(&self, _bs: &Node) -> Result<Option<ImageInfoSpecific>> {
        let s = self.state();
        let bitmaps = s.bitmap_info_list()?;
        let bitmaps = (!bitmaps.is_empty()).then_some(bitmaps);
        let mut data = if s.qcow_version == 2 {
            ImageInfoSpecificQCow2 {
                compat: "0.10".to_string(),
                refcount_bits: s.refcount_bits as i64,
                compression_type: Qcow2CompressionType::Zlib,
                ..ImageInfoSpecificQCow2::default()
            }
        } else {
            ImageInfoSpecificQCow2 {
                compat: "1.1".to_string(),
                lazy_refcounts: Some(s.compatible_features & QCOW2_COMPAT_LAZY_REFCOUNTS != 0),
                corrupt: Some(s.incompatible_features & QCOW2_INCOMPAT_CORRUPT != 0),
                extended_l2: Some(s.has_subclusters()),
                refcount_bits: s.refcount_bits as i64,
                bitmaps,
                data_file: s.image_data_file.clone(),
                data_file_raw: s.has_data_file().then(|| s.data_file_is_raw()),
                compression_type: Qcow2CompressionType::Zlib,
                ..ImageInfoSpecificQCow2::default()
            }
        };
        if let Some(c) = &s.crypto {
            let u = if s.crypt_method_header == QCOW_CRYPT_LUKS {
                c.luks_info().map(ImageInfoSpecificQCow2EncryptionU::Luks)
            } else {
                Some(ImageInfoSpecificQCow2EncryptionU::Aes)
            };
            data.encrypt = u.map(|u| ImageInfoSpecificQCow2Encryption { u });
        }
        Ok(Some(ImageInfoSpecific {
            u: ImageInfoSpecificU::Qcow2(ImageInfoSpecificQCow2Wrapper { data }),
        }))
    }

    /// `qcow2_has_zero_init()`.
    fn has_zero_init(&self, bs: &Node) -> Option<bool> {
        {
            let s = self.state();
            let preallocated = s.l1_size > 0 && s.l1_table.first().is_some_and(|e| *e != 0);
            if !preallocated {
                return Some(true);
            }
            if s.crypto.is_some() {
                return Some(false);
            }
        }
        let data = bs.child("data-file").map_or_else(|| bs.file(), |c| c.node);
        Some(data.has_zero_init())
    }

    /// `qcow2_co_check()`.
    fn check(&self, bs: &Node, fix: u32) -> Option<Result<CheckResult>> {
        let mut res = check::CheckResult::default();
        let r = self.with_bitmaps(bs, false, |s| s.check(&mut res, fix));
        Some(r.map_err(errno_err).map(|()| CheckResult {
            corruptions: res.corruptions,
            leaks: res.leaks,
            check_errors: res.check_errors,
            corruptions_fixed: res.corruptions_fixed,
            leaks_fixed: res.leaks_fixed,
            image_end_offset: res.image_end_offset,
            bfi: crate::node::BlockFragInfo {
                allocated_clusters: res.bfi.allocated_clusters,
                total_clusters: res.bfi.total_clusters,
                fragmented_clusters: res.bfi.fragmented_clusters,
                compressed_clusters: res.bfi.compressed_clusters,
            },
        }))
    }

    /// `qcow2_co_change_backing_file()`.
    fn change_backing_file(
        &self,
        bs: &Node,
        file: Option<&str>,
        format: Option<&str>,
    ) -> Option<std::io::Result<()>> {
        Some(self.lock(bs).change_backing_file(file, format))
    }

    /// `qcow2_make_empty()`.
    fn make_empty(&self, bs: &Node) -> Option<std::io::Result<()>> {
        Some(self.lock(bs).make_empty())
    }

    /// `qcow2_snapshot_create()`.
    fn snapshot_create(&self, bs: &Node, sn: &SnapshotEntry) -> Option<Result<()>> {
        let mut info = SnapshotInfo {
            id_str: sn.id_str.clone(),
            name: sn.name.clone(),
            vm_state_size: sn.vm_state_size,
            date_sec: sn.date_sec,
            date_nsec: sn.date_nsec,
            vm_clock_nsec: sn.vm_clock_nsec,
            icount: sn.icount,
        };
        Some(self.lock(bs).snapshot_create(&mut info).map_err(errno_err))
    }

    /// `qcow2_snapshot_goto()`.
    fn snapshot_goto(&self, bs: &Node, id: &str) -> Option<Result<()>> {
        let old = self.state().total_size;
        let r = self.with_bitmaps(bs, false, |s| s.snapshot_goto(id));
        self.size_changed(bs, old);
        Some(r.map_err(errno_err))
    }

    /// `qcow2_snapshot_delete()`.
    fn snapshot_delete(
        &self,
        bs: &Node,
        id: Option<&str>,
        name: Option<&str>,
    ) -> Option<Result<()>> {
        Some(self.lock(bs).snapshot_delete(id, name))
    }

    /// `qcow2_snapshot_list()`.
    fn snapshot_list(&self, _bs: &Node) -> Option<Result<Vec<SnapshotEntry>>> {
        let r = self.state().snapshot_list();
        Some(r.map_err(errno_err).map(|l| {
            l.into_iter()
                .map(|i| SnapshotEntry {
                    id_str: i.id_str,
                    name: i.name,
                    vm_state_size: i.vm_state_size,
                    date_sec: i.date_sec,
                    date_nsec: i.date_nsec,
                    vm_clock_nsec: i.vm_clock_nsec,
                    icount: i.icount,
                })
                .collect()
        }))
    }

    /// `qcow2_snapshot_load_tmp()`.
    fn snapshot_load_tmp(
        &self,
        bs: &Node,
        id: Option<&str>,
        name: Option<&str>,
    ) -> Option<Result<()>> {
        Some(self.lock(bs).snapshot_load_tmp(id, name))
    }

    fn load_vmstate(&self, bs: &Node, pos: u64, buf: &mut [u8]) -> Option<std::io::Result<()>> {
        Some(self.lock(bs).load_vmstate(pos, buf))
    }

    fn save_vmstate(&self, bs: &Node, pos: u64, buf: &[u8]) -> Option<std::io::Result<()>> {
        Some(self.lock(bs).save_vmstate(pos, buf))
    }

    /// `qcow2_co_invalidate_cache()`.
    fn invalidate_cache(&self, bs: &Node) -> Result<()> {
        let was_inactive = self.state().flags.inactive;
        if !was_inactive {
            let _ = self.do_inactivate(bs);
        }
        // Bitmaps still on the node (the inconsistent ones) are not loaded again.
        let live = node_bitmaps(bs, false);
        let nlive = live.len();
        let mut s = self.lock(bs);
        let crypto = s.crypto.take();
        let mut flags = s.flags;
        flags.inactive = false;
        let mut new = State::blank(s.file.clone(), flags);
        let mut oo = s.options.clone();
        oo.flags = flags;
        oo.data_file = s.data_file.clone();
        new.bitmaps = live;
        let mut no_data_file = |_: &str| -> Result<Arc<dyn Storage>> {
            Err(Error::generic("'data-file' is required for this image"))
        };
        if let Err(e) = new.do_open(&oo, &mut no_data_file) {
            s.drv_gone = true;
            return Err(e.prepend("Could not reopen qcow2 layer: "));
        }
        new.crypto = crypto;
        let loaded = new.bitmaps.split_off(nlive);
        new.bitmaps.clear();
        let backing_file = new.backing_file.clone();
        let backing_format = new.backing_format.clone();
        *s = new;
        drop(s);
        *self.pending.lock().unwrap_or_else(|e| e.into_inner()) = loaded;
        self.publish_bitmaps(bs);
        let mut meta = bs.meta.lock().unwrap_or_else(|e| e.into_inner());
        meta.backing_file = backing_file;
        meta.backing_format = backing_format;
        Ok(())
    }

    /// `qcow2_inactivate()`.
    fn inactivate(&self, bs: &Node) -> Result<()> {
        let r = self.do_inactivate(bs);
        self.state().flags.inactive = true;
        r.map_err(errno_err)?;
        for bm in bs.dirty_bitmaps() {
            if bm.get_persistence() && !bm.inconsistent() {
                bs.release_dirty_bitmap(&bm);
            }
        }
        Ok(())
    }

    fn supported_truncate_flags(&self) -> u32 {
        BDRV_REQ_ZERO_WRITE
    }

    /// `qcow2_co_truncate()`.
    fn truncate_full(
        &self,
        bs: &Node,
        offset: u64,
        exact: bool,
        prealloc: PreallocMode,
        flags: u32,
    ) -> Result<()> {
        self.with_bitmaps(bs, false, |s| {
            s.truncate_flags(offset, exact, prealloc, flags & BDRV_REQ_ZERO_WRITE != 0)
        })
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    /// `qcow2_co_amend()`: `x-blockdev-amend`.
    fn amend(&self, bs: &Node, opts: &BlockdevAmendOptions, force: bool) -> Option<Result<()>> {
        let BlockdevAmendOptionsU::Qcow2(o) = &opts.u else {
            return Some(Err(Error::generic("Driver does not support amending options")));
        };
        let encrypt = o.encrypt.as_ref().map(|e| match &e.u {
            QCryptoBlockAmendOptionsU::Luks(l) => {
                let mut v: EncryptionOptions = vec![("state".into(), l.state.as_str().into())];
                let strs = [
                    ("new-secret", &l.new_secret),
                    ("old-secret", &l.old_secret),
                    ("secret", &l.secret),
                ];
                for (k, s) in strs {
                    if let Some(s) = s {
                        v.push((k.into(), s.clone()));
                    }
                }
                if let Some(k) = l.keyslot {
                    v.push(("keyslot".into(), k.to_string()));
                }
                if let Some(t) = l.iter_time {
                    v.push(("iter-time".into(), t.to_string()));
                }
                (true, v)
            }
            QCryptoBlockAmendOptionsU::Qcow => (false, Vec::new()),
        });
        let mut s = self.lock(bs);
        Some(s.blockdev_amend(encrypt.as_ref().map(|(l, v)| (*l, v)), force))
    }
}

impl Qcow2 {
    fn reopen_prepare_impl(&self, bs: &Node, state: &mut ReopenState) -> Result<()> {
        state.opaque = None;
        let flags = open_flags(&state.flags);
        let mut oo = OpenOptions { flags, ..OpenOptions::default() };
        let mut flat = Vec::new();
        flatten("", &state.options, &mut flat);
        for (k, v) in &flat {
            let known = MUTABLE_OPTS.contains(&k.as_str()) || k.starts_with("encrypt.");
            if known && oo.set(k, v)? {
                state.options.remove(k);
            }
        }
        // Nested forms of the same options are taken as a whole.
        for k in ["encrypt", "overlap-check"] {
            if state.options.get(k).is_some_and(|v| v.as_dict().is_some()) {
                state.options.remove(k);
            }
        }
        if oo.pass_discard_request.is_none() {
            oo.pass_discard_request = Some(flags.unmap);
        }

        let prepared = {
            let mut s = self.lock(bs);
            let old = &s.options;
            oo.encryption = old.encryption.clone();
            oo.data_file = old.data_file.clone();
            if oo.encrypt_format.is_none() && oo.encrypt_opts.is_empty() {
                oo.encrypt_format = old.encrypt_format.clone();
                oo.encrypt_opts = old.encrypt_opts.clone();
            }
            s.update_options_prepare(&oo)?
        };

        if !flags.read_write {
            // qcow2_reopen_bitmaps_ro()
            self.with_bitmaps(bs, true, |s| s.store_persistent_dirty_bitmaps(false))?;
            for bm in bs.dirty_bitmaps() {
                if bm.get_persistence() {
                    bm.set_readonly(true);
                }
            }
            let failed = || {
                Error::generic(format!(
                    "failed while preparing to reopen image '{}'",
                    bs.filename().unwrap_or_default()
                ))
            };
            bs.flush().map_err(|_| failed())?;
            self.lock(bs).mark_clean().map_err(|_| failed())?;
        }

        state.opaque = Some(Box::new(Reopen { prepared, options: oo, flags }));
        Ok(())
    }
}

/// The options of `d` as flat `key=value` text, nested dictionaries joined with dots.
fn flatten(prefix: &str, d: &QDict, out: &mut Vec<(String, String)>) {
    for (k, v) in d.iter() {
        let key = format!("{prefix}{k}");
        match v {
            QValue::Dict(sub) => flatten(&format!("{key}."), sub, out),
            v => out.push((key, value_str(v))),
        }
    }
}

impl PersistentBitmaps for Qcow2 {
    fn supports_persistent_dirty_bitmap(&self, _bs: &Node) -> bool {
        self.state().qcow_version >= 3
    }

    fn can_store_new_dirty_bitmap(&self, bs: &Node, name: &str, granularity: u32) -> Result<()> {
        self.with_bitmaps(bs, false, |s| s.can_store_new_dirty_bitmap(name, granularity, &bs.name))
    }

    fn remove_persistent_dirty_bitmap(&self, bs: &Node, name: &str) -> Result<()> {
        self.lock(bs).remove_persistent_dirty_bitmap(name)
    }
}

/// The input image of `qemu-img measure`.
struct MeasureNode<'a>(&'a Node);

impl MeasureSource for MeasureNode<'_> {
    fn len(&self) -> std::io::Result<u64> {
        self.0.getlength()
    }

    fn block_status(&self, offset: u64, bytes: u64) -> std::io::Result<(u32, u64)> {
        let st = self.0.arc().block_status_above(None, offset, bytes)?;
        Ok((st.ret, st.pnum))
    }

    fn bitmaps_size(&self, cluster_size: u64) -> Option<u64> {
        if !self.0.supports_persistent_dirty_bitmap() {
            return None;
        }
        let list: Vec<(String, u32, u64)> = self
            .0
            .dirty_bitmaps()
            .into_iter()
            .filter(|b| b.get_persistence())
            .map(|b| (b.name().unwrap_or_default(), b.granularity(), b.size()))
            .collect();
        Some(persistent_bitmaps_size(&list, cluster_size))
    }
}

/// `qcow2_measure()`. The options are read in QEMU's order, so the first bad one is the one
/// reported.
fn qcow2_measure(opts: &mut QDict, in_bs: Option<&Node>) -> Result<BlockMeasureInfo> {
    let mut o = CreateOptions::default();
    let extended_l2 = take_bool(opts, "extended_l2")?.unwrap_or(false);
    o.extended_l2 = Some(extended_l2);
    let cluster_size = take_size(opts, "cluster_size")?.unwrap_or(DEFAULT_CLUSTER_SIZE);
    validate_cluster_size(cluster_size, extended_l2)?;
    o.cluster_size = Some(cluster_size);
    let version = match take_str(opts, "compat").as_deref() {
        None | Some("1.1") => 3,
        Some("0.10") => 2,
        Some(v) => return Err(Error::generic(format!("Invalid compatibility level: '{v}'"))),
    };
    o.version = Some(version);
    if let Some(v) = take_str(opts, "refcount_bits") {
        o.set("refcount_bits", &v)?;
    }
    let refcount_bits = o.refcount_bits.unwrap_or(16);
    if refcount_bits > 64 || !refcount_bits.is_power_of_two() {
        return Err(Error::generic(
            "Refcount width must be a power of two and may not exceed 64 bits",
        ));
    }
    if version < 3 && refcount_bits != 16 {
        return Err(Error::generic(
            "Different refcount widths than 16 bits require compatibility level 1.1 or above \
             (use compat=1.1 or greater)",
        ));
    }
    if let Some(p) = take_str(opts, "preallocation") {
        o.preallocation = Some(
            resize::parse_prealloc(&p)
                .map_err(|_| Error::generic(format!("invalid parameter value: {p}")))?,
        );
    }
    o.backing_file = take_str(opts, "backing_file");
    if take_str(opts, "encrypt.format").as_deref() == Some("luks") {
        o.encrypt_format = Some(EncryptionFormat::Luks);
        let keys: Vec<String> =
            opts.keys().filter(|k| k.starts_with("encrypt.")).map(str::to_string).collect();
        for k in keys {
            if let Some(v) = take_str(opts, &k) {
                o.encrypt_opts.push((k["encrypt.".len()..].to_string(), v));
            }
        }
        o.encryption = Some(Arc::new(CryptoProvider));
    }
    o.size = take_size(opts, "size")?.unwrap_or(0);
    let src = in_bs.map(MeasureNode);
    create::measure(&o, src.as_ref().map(|s| s as &dyn MeasureSource))
}

/// `qcow2_amend_options()`: `qemu-img amend`. Options qcow2 does not know stay in `options`
/// for the caller to report.
fn qcow2_amend_options(bs: &Node, options: &mut QDict, force: bool) -> Result<()> {
    let Some(d) = bs.driver.as_any().and_then(|a| a.downcast_ref::<Qcow2>()) else {
        return Err(Error::generic("Driver does not support amending options"));
    };
    let mut o = amend::AmendOptions::default();
    let keys: Vec<String> = options.keys().map(str::to_string).collect();
    for k in keys {
        let Some(v) = options.get(&k).map(value_str) else { continue };
        if o.set(&k, &v)? {
            options.remove(&k);
        }
    }
    let old = d.state().total_size;
    let r = d.with_bitmaps(bs, true, |s| s.amend_options(&o, &mut |_, _| {}, force));
    d.size_changed(bs, old);
    r
}

/// Deletes a file `qcow2_co_create_opts()` made, `bdrv_co_delete_file_noerr()`.
fn delete_file_noerr(node: &Node) {
    if node.driver_name != "file" {
        return;
    }
    let Some(name) = node.filename() else { return };
    if let Err(e) = std::fs::remove_file(&name) {
        error_report(Error::from_io(format!("Failed to delete file '{name}'"), e).message());
    }
}

/// The typed encryption options as the text `encrypt.*` options.
fn create_encrypt_opts(e: &QCryptoBlockCreateOptions) -> (EncryptionFormat, EncryptionOptions) {
    let mut v: EncryptionOptions = Vec::new();
    match &e.u {
        QCryptoBlockCreateOptionsU::Qcow(q) => {
            if let Some(k) = &q.key_secret {
                v.push(("key-secret".into(), k.clone()));
            }
            (EncryptionFormat::Aes, v)
        }
        QCryptoBlockCreateOptionsU::Luks(l) => {
            if let Some(k) = &l.key_secret {
                v.push(("key-secret".into(), k.clone()));
            }
            let enums = [
                ("cipher-alg", l.cipher_alg.map(|x| x.as_str())),
                ("cipher-mode", l.cipher_mode.map(|x| x.as_str())),
                ("ivgen-alg", l.ivgen_alg.map(|x| x.as_str())),
                ("ivgen-hash-alg", l.ivgen_hash_alg.map(|x| x.as_str())),
                ("hash-alg", l.hash_alg.map(|x| x.as_str())),
            ];
            for (k, s) in enums {
                if let Some(s) = s {
                    v.push((k.into(), s.to_string()));
                }
            }
            if let Some(t) = l.iter_time {
                v.push(("iter-time".into(), t.to_string()));
            }
            (EncryptionFormat::Luks, v)
        }
    }
}

/// `qcow2_co_create()`: `blockdev-create` for qcow2.
fn qcow2_co_create(graph: &BlockGraph, options: BlockdevCreateOptionsU) -> Result<()> {
    let BlockdevCreateOptionsU::Qcow2(o) = options else {
        unreachable!("qcow2 driver with other create options")
    };
    do_create(graph, o)
}

fn do_create(graph: &BlockGraph, o: BlockdevCreateOptionsQcow2) -> Result<()> {
    let mut c = CreateOptions::default();
    c.size = o.size;
    c.version = o.version.map(|v| match v {
        BlockdevQcow2Version::V2 => 2,
        BlockdevQcow2Version::V3 => 3,
    });
    c.cluster_size = o.cluster_size;
    c.extended_l2 = o.extended_l2;
    c.preallocation = o.preallocation;
    c.lazy_refcounts = o.lazy_refcounts;
    c.refcount_bits = o.refcount_bits.map(|b| b as u64);
    c.compression_type = o.compression_type.map(|t| match t {
        Qcow2CompressionType::Zlib => state::QCOW2_COMPRESSION_TYPE_ZLIB,
    });
    c.backing_file = o.backing_file;
    c.backing_fmt = o.backing_fmt.map(|f| f.as_str().to_string());
    c.data_file_raw = o.data_file_raw;
    c.encryption = Some(Arc::new(CryptoProvider));
    if let Some(e) = &o.encrypt {
        let (f, v) = create_encrypt_opts(e);
        c.encrypt_format = Some(f);
        c.encrypt_opts = v;
    }
    let blk = graph.open_create_blk(o.file)?;
    let node = blk.root().expect("a new backend has its node");
    let data_blk = match o.data_file {
        Some(r) => Some(graph.open_create_blk(r)?),
        None => None,
    };
    let data = data_blk.as_ref().map(|b| {
        Arc::new(NodeStorage(b.root().expect("a new backend has its node"))) as Arc<dyn Storage>
    });
    create::create(Arc::new(NodeStorage(node)), data, &c)
}

/// The `-o` names `qcow2_co_create_opts()` takes for itself, `qcow2_create_opts`.
fn create_opt_names() -> impl Iterator<Item = &'static str> {
    create::CREATE_OPTS.iter().map(|d| d.name)
}

/// `qcow2_co_create_opts()`: `qemu-img create -f qcow2`.
fn qcow2_co_create_opts(filename: &str, options: &mut QDict) -> Result<()> {
    let mut qdict = QDict::new();
    for name in create_opt_names() {
        if let Some(v) = take_str(options, name) {
            qdict.put(name, v);
        }
    }

    // Handle encryption options.
    match qdict.get_str("encryption") {
        Some("on") => qdict.put("encryption", "qcow"),
        Some("off") => {
            qdict.remove("encryption");
        }
        _ => {}
    }
    if qdict.get_str("encrypt.format") == Some("aes") {
        qdict.put("encrypt.format", "qcow");
    }

    // Convert compat=0.10/1.1 into compat=v2/v3, to be renamed into version=v2/v3 below.
    match qdict.get_str("compat") {
        Some("0.10") => qdict.put("compat", "v2"),
        Some("1.1") => qdict.put("compat", "v3"),
        _ => {}
    }

    let mut keep_data_file = false;
    if let Some(v) = qdict.get_str("keep_data_file").map(str::to_string) {
        keep_data_file = match v.as_str() {
            "on" => true,
            "off" => false,
            _ => {
                return Err(Error::generic(format!(
                    "Invalid value '{v}' for 'keep_data_file': Must be 'on' or 'off'"
                )));
            }
        };
        qdict.remove("keep_data_file");
    }

    // Change legacy command line options into QMP ones.
    rename_keys(
        &mut qdict,
        &[
            ("backing_file", "backing-file"),
            ("backing_fmt", "backing-fmt"),
            ("cluster_size", "cluster-size"),
            ("lazy_refcounts", "lazy-refcounts"),
            ("extended_l2", "extended-l2"),
            ("refcount_bits", "refcount-bits"),
            ("encryption", "encrypt.format"),
            ("compat", "version"),
            ("data_file_raw", "data-file-raw"),
            ("compression_type", "compression-type"),
        ],
    )?;

    // Create and open the file (protocol layer).
    let mut data_options = options.clone();
    let (graph, blk, mut all) = create_opts_open(filename, options, "qcow2", &[], &[])?;
    let image = blk.root().expect("a new backend has its node");

    let r = (|| -> Result<(bool, Option<Arc<Node>>)> {
        // Create and open an external data file (protocol layer).
        let mut data_node = None;
        if let Some(val) = qdict.get_str("data_file").map(str::to_string) {
            if !keep_data_file {
                graph.create_file(&val, &mut data_options)?;
            }
            let data_blk = graph.open_protocol_blk(&val)?;
            let n = data_blk.root().expect("a new backend has its node");
            qdict.remove("data_file");
            qdict.put("data-file", n.name.as_str());
            data_node = Some(n);
            // The node stays in the graph by name; the backend can go.
            std::mem::forget(data_blk);
        } else if keep_data_file {
            return Err(Error::generic("Must not use 'keep_data_file=on' without 'data_file'"));
        }
        Ok((false, data_node))
    })();
    let data_node = match r {
        Ok((_, n)) => n,
        Err(e) => {
            delete_file_noerr(&image);
            return Err(e);
        }
    };

    let r = (|| -> Result<()> {
        for (k, v) in qdict.iter() {
            all.put(k, v.clone());
        }
        let create_options = visit_create_options(all)?;
        let BlockdevCreateOptionsU::Qcow2(mut o) = create_options.u else {
            unreachable!("driver is qcow2")
        };
        let prealloc = o.preallocation.unwrap_or(PreallocMode::Off);
        o.preallocation = Some(prealloc);
        if keep_data_file && prealloc != PreallocMode::Off && prealloc != PreallocMode::Metadata {
            return Err(Error::generic(
                "Preallocating more than only metadata would overwrite the external data file's \
                 content and is therefore incompatible with 'keep_data_file=on'",
            ));
        }
        if keep_data_file && prealloc == PreallocMode::Off && !o.data_file_raw.unwrap_or(false) {
            return Err(Error::generic(
                "'keep_data_file=on' requires 'preallocation=metadata' or 'data_file_raw=on', or \
                 the file contents will not be visible",
            ));
        }
        // Silently round up the size.
        o.size = o.size.div_ceil(BDRV_SECTOR_SIZE) * BDRV_SECTOR_SIZE;
        debug_assert!(matches!(o.file, BlockdevRef::Reference(_)));
        do_create(&graph, o)
    })();
    if r.is_err() {
        delete_file_noerr(&image);
        if !keep_data_file {
            if let Some(n) = &data_node {
                delete_file_noerr(n);
            }
        }
    }
    r
}

/// `qdict_rename_keys()`.
fn rename_keys(qdict: &mut QDict, renames: &[(&str, &str)]) -> Result<()> {
    for (from, to) in renames {
        if let Some(v) = qdict.get(from).cloned() {
            if qdict.contains_key(to) {
                return Err(Error::generic(format!(
                    "'{to}' and its alias '{from}' can't be used at the same time"
                )));
            }
            qdict.put(*to, v);
            qdict.remove(from);
        }
    }
    Ok(())
}
