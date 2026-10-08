// SPDX-License-Identifier: GPL-2.0-or-later

//! `BlockBackend` from block/block-backend.c: what a guest device or an export holds to get at a
//! node, with the permissions it needs.
//!
//! The I/O methods are synchronous for now and run on the caller's thread. The `blk_aio_*()`
//! family and the coroutine versions come with the ruvm-aio integration; they will sit on top
//! of the same checks.

use std::io;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use ruvm_base::{Error, Result};

use crate::accounting::{BlockAcctStats, BlockAcctType};
use crate::drain::{aio_wait_kick, aio_wait_while};
use crate::graph::BlockGraph;
use crate::node::{ENOMEDIUM, Node, Parent, ParentOps, errno, new_edge_id};
use crate::perm::{BLK_PERM_WRITE, BLK_PERM_WRITE_UNCHANGED};
use crate::throttle::groups::ThrottleGroupMember;
use crate::throttle::{ThrottleConfig, ThrottleDirection};

/// `BDRV_REQUEST_MAX_BYTES`: the largest single request, `MIN(SIZE_MAX, INT_MAX)` aligned down.
const BDRV_REQUEST_MAX_BYTES: u64 = (i32::MAX as u64) & !511;

#[derive(Debug)]
struct State {
    perm: u64,
    shared: u64,
    /// `blk->dev`, here the id of the device, or an empty string for a device without one.
    dev: Option<String>,
}

/// The drain side of a backend: `blk->quiesce_counter`, `blk->in_flight` and the queue of
/// requests waiting for a drained section to end. It is the backend's `BdrvChildClass` as the
/// root node sees it.
#[derive(Debug, Default)]
pub(crate) struct BlkIo {
    quiesce_counter: AtomicU32,
    in_flight: AtomicU32,
    /// `blk->disable_request_queuing`.
    disable_request_queuing: AtomicBool,
    queue: Mutex<()>,
    queue_cond: Condvar,
    /// `blk->public.throttle_group_member`, in a group once `-drive throttling.*` or
    /// [`BlockBackend::io_limits_enable`] put it there.
    tgm: ThrottleGroupMember,
    /// `blk->root`: the node, or `None` for an empty drive. It sits here rather than in the
    /// backend so that `bdrv_replace_node()` can move the edge through [`ParentOps`].
    root: Mutex<Option<Arc<Node>>>,
}

impl BlkIo {
    /// `blk_inc_in_flight()`.
    fn inc_in_flight(&self) {
        self.in_flight.fetch_add(1, Ordering::SeqCst);
    }

    /// `blk_dec_in_flight()`.
    fn dec_in_flight(&self) {
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        aio_wait_kick();
    }

    /// `blk_wait_while_drained()`: a new request waits while the backend is drained, unless
    /// queuing is disabled. The caller holds an in-flight reference, which is given up while
    /// waiting so the drain can finish.
    fn wait_while_drained(&self) {
        if self.disable_request_queuing.load(Ordering::SeqCst) {
            return;
        }
        let mut g = self.queue.lock().unwrap();
        while self.quiesce_counter.load(Ordering::SeqCst) > 0 {
            self.dec_in_flight();
            g = self.queue_cond.wait(g).unwrap();
            self.inc_in_flight();
        }
    }
}

impl ParentOps for BlkIo {
    fn drained_begin(&self) {
        self.quiesce_counter.fetch_add(1, Ordering::SeqCst);
        self.tgm.io_limits_disable_begin();
    }

    fn drained_end(&self) {
        let _g = self.queue.lock().unwrap();
        let old = self.quiesce_counter.fetch_sub(1, Ordering::SeqCst);
        assert!(old > 0, "unbalanced drained_end on a block backend");
        self.tgm.io_limits_disable_end();
        if old == 1 {
            self.queue_cond.notify_all();
        }
    }

    fn drained_poll(&self) -> bool {
        self.in_flight.load(Ordering::SeqCst) > 0
    }

    fn set_node(&self, node: Arc<Node>) {
        *self.root.lock().unwrap() = Some(node);
    }

    fn is_backend(&self) -> bool {
        true
    }
}

/// An in-flight reference on a backend for the length of one request.
struct InFlight<'a>(&'a BlkIo);

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        self.0.dec_in_flight();
    }
}

/// A `BlockBackend`.
#[derive(Debug)]
pub struct BlockBackend {
    pub(crate) io: Arc<BlkIo>,
    /// `blk->name`, set for backends the monitor knows by name, like the ones `-drive` makes.
    name: Option<String>,
    /// The edge id of `blk->root`.
    edge: u64,
    state: Mutex<State>,
    enable_write_cache: AtomicBool,
    /// `blk->root_state.read_only`, what an empty drive reports.
    empty_read_only: bool,
    /// `blk->stats`.
    stats: BlockAcctStats,
}

impl BlockBackend {
    /// An anonymous backend on the node `node_name`, or on the root node of the backend of
    /// that name, holding `perm` and sharing `shared`. A guest disk attaches with
    /// `BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE` and shares `BLK_PERM_CONSISTENT_READ |
    /// BLK_PERM_WRITE_UNCHANGED`, as `blkconf_apply_backend_options()` does.
    pub fn new(graph: &BlockGraph, node_name: &str, perm: u64, shared: u64) -> Result<Arc<Self>> {
        let node = graph.lookup_bs(node_name)?;
        Ok(Arc::new(Self::with_node(None, node, perm, shared)?))
    }

    /// `blk_new()`: a backend with no medium. `perm` and `shared` apply once a node is
    /// inserted.
    pub(crate) fn new_empty(name: Option<String>, perm: u64, shared: u64, read_only: bool) -> Self {
        BlockBackend {
            io: Arc::new(BlkIo::default()),
            name,
            edge: new_edge_id(),
            state: Mutex::new(State { perm, shared, dev: None }),
            enable_write_cache: AtomicBool::new(true),
            empty_read_only: read_only,
            stats: BlockAcctStats::default(),
        }
    }

    /// `blk_new_with_bs()`: a backend on `node`, taking `perm` on it and sharing `shared`.
    pub(crate) fn with_node(
        name: Option<String>,
        node: Arc<Node>,
        perm: u64,
        shared: u64,
    ) -> Result<Self> {
        let blk = Self::new_empty(name, perm, shared, false);
        blk.insert(node)?;
        Ok(blk)
    }

    /// `blk_insert_bs()`.
    pub(crate) fn insert(&self, node: Arc<Node>) -> Result<()> {
        let s = self.state.lock().unwrap();
        let mut root = self.io.root.lock().unwrap();
        assert!(root.is_none(), "backend already has a medium");
        node.update_parent(self.edge, Some(self.parent(&s, s.perm, s.shared)))?;
        *root = Some(node);
        Ok(())
    }

    fn parent(&self, s: &State, perm: u64, shared: u64) -> Parent {
        // blk_root_get_parent_desc().
        let desc = match (&self.name, &s.dev) {
            (Some(n), _) => format!("block device '{n}'"),
            (None, Some(d)) if !d.is_empty() => format!("block device '{d}'"),
            _ => "an unnamed block device".to_string(),
        };
        Parent {
            id: self.edge,
            desc,
            child_name: "root".to_string(),
            perm,
            shared,
            ops: Some(self.io.clone()),
            quiesced: false,
        }
    }

    /// Starts a request: takes an in-flight reference and waits out a drained section.
    fn begin(&self) -> InFlight<'_> {
        self.io.inc_in_flight();
        self.io.wait_while_drained();
        InFlight(&self.io)
    }

    /// `blk_set_disable_request_queuing()`.
    pub fn set_disable_request_queuing(&self, disable: bool) {
        self.io.disable_request_queuing.store(disable, Ordering::SeqCst);
    }

    /// `blk->in_flight`.
    pub fn in_flight(&self) -> u32 {
        self.io.in_flight.load(Ordering::SeqCst)
    }

    /// `blk->quiesce_counter`.
    pub fn quiesce_counter(&self) -> u32 {
        self.io.quiesce_counter.load(Ordering::SeqCst)
    }

    /// `blk_drain()`: waits until no request of this backend is in flight, draining the root
    /// node too.
    pub fn drain(&self) {
        if let Some(root) = self.root() {
            root.drained_begin();
            aio_wait_while(|| self.io.in_flight.load(Ordering::SeqCst) > 0);
            root.drained_end();
        } else {
            aio_wait_while(|| self.io.in_flight.load(Ordering::SeqCst) > 0);
        }
    }

    /// `blk_bs()`.
    pub(crate) fn root(&self) -> Option<Arc<Node>> {
        self.io.root.lock().unwrap().clone()
    }

    /// `blk_name()`: the name, or an empty string for an anonymous backend.
    pub fn name(&self) -> &str {
        self.name.as_deref().unwrap_or("")
    }

    /// `bdrv_get_node_name(blk_bs())`, `None` for an empty drive.
    pub fn node_name(&self) -> Option<String> {
        self.root().map(|n| n.name.clone())
    }

    /// `blk_set_perm()`: change what the backend holds and shares. On failure nothing changes.
    pub fn set_perm(&self, perm: u64, shared: u64) -> Result<()> {
        let mut s = self.state.lock().unwrap();
        if let Some(root) = self.root() {
            // bdrv_child_try_set_perm()
            root.try_set_parent_perm(self.edge, perm, shared)?;
        }
        s.perm = perm;
        s.shared = shared;
        Ok(())
    }

    /// `blk_get_perm()`: the permissions held, and those shared.
    pub fn perm(&self) -> (u64, u64) {
        let s = self.state.lock().unwrap();
        (s.perm, s.shared)
    }

    /// `blk_attach_dev()`, with the device's id (`""` for a device without one). Fails with
    /// `EBUSY` if a device is already attached.
    pub fn attach_dev(&self, dev_id: &str) -> io::Result<()> {
        let mut s = self.state.lock().unwrap();
        if s.dev.is_some() {
            return Err(errno(libc::EBUSY));
        }
        s.dev = Some(dev_id.to_string());
        // Error messages name the device from now on.
        if let Some(root) = self.root() {
            let p = self.parent(&s, s.perm, s.shared);
            let _ = root.update_parent(self.edge, Some(p));
        }
        Ok(())
    }

    /// `blk_detach_dev()`: the device lets go, and so do its permissions, as
    /// `blk_set_perm(blk, 0, BLK_PERM_ALL)` there.
    pub fn detach_dev(&self) {
        let mut s = self.state.lock().unwrap();
        s.dev = None;
        s.perm = 0;
        s.shared = crate::perm::BLK_PERM_ALL;
        if let Some(root) = self.root() {
            let p = self.parent(&s, s.perm, s.shared);
            let _ = root.update_parent(self.edge, Some(p));
        }
    }

    /// The id of the attached device, if any.
    pub fn attached_dev(&self) -> Option<String> {
        self.state.lock().unwrap().dev.clone()
    }

    /// `blk_is_inserted()`: whether there is a medium.
    pub fn is_inserted(&self) -> bool {
        self.root().is_some()
    }

    /// `blk_is_read_only()`.
    pub fn is_read_only(&self) -> bool {
        self.root().map_or(self.empty_read_only, |n| n.read_only())
    }

    /// `blk_is_writable()`.
    pub fn is_writable(&self) -> bool {
        !self.is_read_only()
    }

    /// `blk_enable_write_cache()`: false means every write is followed by a flush.
    pub fn enable_write_cache(&self) -> bool {
        self.enable_write_cache.load(Ordering::Relaxed)
    }

    /// `blk_set_enable_write_cache()`, what a guest toggles through the device.
    pub fn set_enable_write_cache(&self, wce: bool) {
        self.enable_write_cache.store(wce, Ordering::Relaxed);
    }

    fn available(&self) -> io::Result<Arc<Node>> {
        self.root().ok_or_else(|| errno(ENOMEDIUM))
    }

    /// `blk_check_byte_request()`: requests stay inside the medium.
    fn check_byte_request(&self, node: &Node, offset: u64, bytes: u64) -> io::Result<()> {
        if bytes > BDRV_REQUEST_MAX_BYTES || offset > i64::MAX as u64 {
            return Err(errno(libc::EIO));
        }
        let len = node.getlength()?;
        if offset > len || bytes > len - offset {
            return Err(errno(libc::EIO));
        }
        Ok(())
    }

    fn check_write(&self, node: &Node) -> io::Result<()> {
        // QEMU asserts this, a device must take the write permission before writing.
        let perm = self.state.lock().unwrap().perm;
        if perm & (BLK_PERM_WRITE | BLK_PERM_WRITE_UNCHANGED) == 0 || node.read_only() {
            return Err(errno(libc::EPERM));
        }
        Ok(())
    }

    /// `blk_co_getlength()`: the size in bytes, a multiple of 512.
    pub fn getlength(&self) -> io::Result<u64> {
        self.available()?.getlength()
    }

    /// `blk_pread()`.
    pub fn pread(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let _r = self.begin();
        let node = self.available()?;
        let bytes = buf.len() as u64;
        self.check_acct(&node, offset, bytes, BlockAcctType::Read)?;
        self.acct(bytes, BlockAcctType::Read, || {
            self.io.tgm.io_limits_intercept(bytes, ThrottleDirection::Read);
            node.pread(offset, buf)
        })
    }

    /// `blk_pwrite()`. With the write cache off the write is `BDRV_REQ_FUA`, done here as a
    /// flush afterwards like `bdrv_driver_pwritev()` does for drivers without FUA.
    pub fn pwrite(&self, offset: u64, buf: &[u8]) -> io::Result<()> {
        let _r = self.begin();
        let node = self.available()?;
        let bytes = buf.len() as u64;
        self.check_acct(&node, offset, bytes, BlockAcctType::Write)?;
        self.check_write(&node)?;
        self.acct(bytes, BlockAcctType::Write, || {
            self.io.tgm.io_limits_intercept(bytes, ThrottleDirection::Write);
            node.pwrite(offset, buf)?;
            self.fua(&node)
        })
    }

    /// `blk_pwrite_compressed()`: writes whole clusters compressed, for `qemu-img convert
    /// -c`.
    pub fn pwrite_compressed(&self, offset: u64, buf: &[u8]) -> io::Result<()> {
        let _r = self.begin();
        let node = self.available()?;
        let bytes = buf.len() as u64;
        self.check_acct(&node, offset, bytes, BlockAcctType::Write)?;
        self.check_write(&node)?;
        self.acct(bytes, BlockAcctType::Write, || {
            self.io.tgm.io_limits_intercept(bytes, ThrottleDirection::Write);
            node.pwrite_compressed(offset, buf)?;
            self.fua(&node)
        })
    }

    /// The request and permission checks of a write, for the whole-node operations that
    /// write without going through the methods here.
    pub(crate) fn check_io_write(&self, node: &Node, offset: u64, bytes: u64) -> io::Result<()> {
        self.check_byte_request(node, offset, bytes)?;
        self.check_write(node)
    }

    fn fua(&self, node: &Node) -> io::Result<()> {
        if self.enable_write_cache() { Ok(()) } else { node.flush() }
    }

    /// `blk_pwrite_zeroes()`. `may_unmap` is `BDRV_REQ_MAY_UNMAP`, only honoured when the node
    /// was opened with `discard=unmap`.
    pub fn pwrite_zeroes(&self, offset: u64, bytes: u64, may_unmap: bool) -> io::Result<()> {
        let _r = self.begin();
        let node = self.available()?;
        self.check_acct(&node, offset, bytes, BlockAcctType::Write)?;
        self.check_write(&node)?;
        self.acct(bytes, BlockAcctType::Write, || {
            self.io.tgm.io_limits_intercept(bytes, ThrottleDirection::Write);
            node.pwrite_zeroes(offset, bytes, may_unmap)?;
            self.fua(&node)
        })
    }

    /// `blk_pdiscard()`: advice, a node without `discard=unmap` ignores it.
    pub fn pdiscard(&self, offset: u64, bytes: u64) -> io::Result<()> {
        let _r = self.begin();
        let node = self.available()?;
        self.check_acct(&node, offset, bytes, BlockAcctType::Unmap)?;
        self.check_write(&node)?;
        self.acct(bytes, BlockAcctType::Unmap, || node.pdiscard(offset, bytes))
    }

    /// `blk_flush()`.
    pub fn flush(&self) -> io::Result<()> {
        let _r = self.begin();
        let node = self.available()?;
        self.acct(0, BlockAcctType::Flush, || node.flush())
    }

    /// `blk_check_byte_request()`, counting a refused request as invalid as the devices do
    /// with `block_acct_invalid()`.
    fn check_acct(
        &self,
        node: &Node,
        offset: u64,
        bytes: u64,
        ty: BlockAcctType,
    ) -> io::Result<()> {
        let r = self.check_byte_request(node, offset, bytes);
        if r.is_err() {
            self.stats.invalid(ty);
        }
        r
    }

    /// Runs `f` between `block_acct_start()` and `block_acct_done()` or `block_acct_failed()`.
    fn acct<T>(
        &self,
        bytes: u64,
        ty: BlockAcctType,
        f: impl FnOnce() -> io::Result<T>,
    ) -> io::Result<T> {
        let mut cookie = self.stats.start(bytes, ty);
        let r = f();
        if r.is_ok() {
            self.stats.done(&mut cookie);
        } else {
            self.stats.failed(&mut cookie);
        }
        r
    }

    /// `blk_get_stats()`, for `stats-account-*`, `stats-intervals` and latency histograms.
    pub fn acct_stats(&self) -> &BlockAcctStats {
        &self.stats
    }

    /// The `BlockDeviceStats` of the backend in `query-blockstats`, with the root node's
    /// `wr_highest_offset`.
    pub fn stats(&self) -> ruvm_qapi::types::BlockDeviceStats {
        let mut s = self.stats.query();
        if let Some(root) = self.root() {
            s.wr_highest_offset = root.io.wr_highest_offset() as i64;
        }
        s
    }

    /// `blk_io_limits_enable()`: puts the backend in the throttle group `group`, which is
    /// made if it does not exist.
    pub fn io_limits_enable(&self, group: &str) {
        assert!(!self.io.tgm.is_registered(), "the backend is in a throttle group already");
        self.io.tgm.register(group);
    }

    /// `blk_io_limits_disable()`: takes the backend out of its throttle group, draining it
    /// first so that no request waits in the group.
    pub fn io_limits_disable(&self) {
        if !self.io.tgm.is_registered() {
            return;
        }
        let root = self.root();
        if let Some(n) = &root {
            n.drained_begin();
        }
        self.io.tgm.unregister();
        if let Some(n) = &root {
            n.drained_end();
        }
    }

    /// `blk_io_limits_update_group()`: moves the backend to the group `group`.
    pub fn io_limits_update_group(&self, group: &str) {
        if !self.io.tgm.is_registered() {
            return;
        }
        if self.io.tgm.group_name().as_deref() == Some(group) {
            return;
        }
        self.io_limits_disable();
        self.io_limits_enable(group);
    }

    /// `blk_set_io_limits()`: the limits of the backend's group.
    pub fn set_io_limits(&self, cfg: &ThrottleConfig) {
        self.io.tgm.config(cfg);
    }

    /// The group of the backend and its limits, what `query-block` shows as `iops`, `bps`
    /// and `group`.
    pub fn io_limits(&self) -> Option<(String, ThrottleConfig)> {
        let name = self.io.tgm.group_name()?;
        Some((name, self.io.tgm.get_config()?))
    }

    /// `blk_truncate()` with `PREALLOC_MODE_OFF`. Needs the `resize` permission.
    pub fn truncate(&self, len: u64) -> Result<()> {
        let Some(node) = self.root() else {
            return Err(Error::generic("No medium inserted"));
        };
        if self.state.lock().unwrap().perm & crate::perm::BLK_PERM_RESIZE == 0 {
            return Err(Error::generic("blk_truncate() needs the 'resize' permission"));
        }
        node.truncate(len)
    }
}

impl Drop for BlockBackend {
    fn drop(&mut self) {
        if let Some(root) = self.io.root.lock().unwrap().take() {
            let _ = root.update_parent(self.edge, None);
        }
    }
}
