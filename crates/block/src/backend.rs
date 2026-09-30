// SPDX-License-Identifier: GPL-2.0-or-later

//! `BlockBackend` from block/block-backend.c: what a guest device or an export holds to get at a
//! node, with the permissions it needs.
//!
//! The I/O methods are synchronous for now and run on the caller's thread. The `blk_aio_*()`
//! family and the coroutine versions come with the ruvm-aio integration; they will sit on top
//! of the same checks.

use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use ruvm_base::{Error, Result};

use crate::graph::BlockGraph;
use crate::node::{ENOMEDIUM, Node, Parent, errno, new_edge_id};
use crate::perm::{BLK_PERM_WRITE, BLK_PERM_WRITE_UNCHANGED};

/// `BDRV_REQUEST_MAX_BYTES`: the largest single request, `MIN(SIZE_MAX, INT_MAX)` aligned down.
const BDRV_REQUEST_MAX_BYTES: u64 = (i32::MAX as u64) & !511;

#[derive(Debug)]
struct State {
    root: Option<Arc<Node>>,
    perm: u64,
    shared: u64,
    /// `blk->dev`, here the id of the device, or an empty string for a device without one.
    dev: Option<String>,
}

/// A `BlockBackend`.
#[derive(Debug)]
pub struct BlockBackend {
    /// `blk->name`, set for backends the monitor knows by name, like the ones `-drive` makes.
    name: Option<String>,
    /// The edge id of `blk->root`.
    edge: u64,
    state: Mutex<State>,
    enable_write_cache: AtomicBool,
    /// `blk->root_state.read_only`, what an empty drive reports.
    empty_read_only: bool,
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
            name,
            edge: new_edge_id(),
            state: Mutex::new(State { root: None, perm, shared, dev: None }),
            enable_write_cache: AtomicBool::new(true),
            empty_read_only: read_only,
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
        let mut s = self.state.lock().unwrap();
        assert!(s.root.is_none(), "backend already has a medium");
        node.update_parent(self.edge, Some(self.parent(&s, s.perm, s.shared)))?;
        s.root = Some(node);
        Ok(())
    }

    fn parent(&self, s: &State, perm: u64, shared: u64) -> Parent {
        // blk_root_get_parent_desc().
        let desc = match (&self.name, &s.dev) {
            (Some(n), _) => format!("block device '{n}'"),
            (None, Some(d)) if !d.is_empty() => format!("block device '{d}'"),
            _ => "an unnamed block device".to_string(),
        };
        Parent { id: self.edge, desc, child_name: "root", perm, shared }
    }

    fn root(&self) -> Option<Arc<Node>> {
        self.state.lock().unwrap().root.clone()
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
        if let Some(root) = &s.root {
            root.update_parent(self.edge, Some(self.parent(&s, perm, shared)))?;
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
        if let Some(root) = s.root.clone() {
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
        if let Some(root) = s.root.clone() {
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
        self.root().map_or(self.empty_read_only, |n| n.flags.read_only)
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
        if perm & (BLK_PERM_WRITE | BLK_PERM_WRITE_UNCHANGED) == 0 || node.flags.read_only {
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
        let node = self.available()?;
        self.check_byte_request(&node, offset, buf.len() as u64)?;
        node.pread(offset, buf)
    }

    /// `blk_pwrite()`. With the write cache off the write is `BDRV_REQ_FUA`, done here as a
    /// flush afterwards like `bdrv_driver_pwritev()` does for drivers without FUA.
    pub fn pwrite(&self, offset: u64, buf: &[u8]) -> io::Result<()> {
        let node = self.available()?;
        self.check_byte_request(&node, offset, buf.len() as u64)?;
        self.check_write(&node)?;
        node.pwrite(offset, buf)?;
        self.fua(&node)
    }

    fn fua(&self, node: &Node) -> io::Result<()> {
        if self.enable_write_cache() { Ok(()) } else { node.flush() }
    }

    /// `blk_pwrite_zeroes()`. `may_unmap` is `BDRV_REQ_MAY_UNMAP`, only honoured when the node
    /// was opened with `discard=unmap`.
    pub fn pwrite_zeroes(&self, offset: u64, bytes: u64, may_unmap: bool) -> io::Result<()> {
        let node = self.available()?;
        self.check_byte_request(&node, offset, bytes)?;
        self.check_write(&node)?;
        node.pwrite_zeroes(offset, bytes, may_unmap)?;
        self.fua(&node)
    }

    /// `blk_pdiscard()`: advice, a node without `discard=unmap` ignores it.
    pub fn pdiscard(&self, offset: u64, bytes: u64) -> io::Result<()> {
        let node = self.available()?;
        self.check_byte_request(&node, offset, bytes)?;
        self.check_write(&node)?;
        node.pdiscard(offset, bytes)
    }

    /// `blk_flush()`.
    pub fn flush(&self) -> io::Result<()> {
        self.available()?.flush()
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
        if let Some(root) = self.state.get_mut().unwrap().root.take() {
            let _ = root.update_parent(self.edge, None);
        }
    }
}
