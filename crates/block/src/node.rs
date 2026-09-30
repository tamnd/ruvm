// SPDX-License-Identifier: GPL-2.0-or-later

//! A live node: its driver, its children, the permissions its parents hold, and the generic
//! request layer from block/io.c that sits between a parent and the driver.
//!
//! Requests are synchronous for now. They run on the caller's thread and block it until the host
//! call returns, which is what QEMU's `aio=threads` does from the point of view of the worker
//! thread. The asynchronous path on ruvm-aio comes later and will keep these semantics.

use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ruvm_base::{Error, Result};
use ruvm_qapi::types::BlockdevDetectZeroesOptions;

use crate::perm::{
    BLK_PERM_ALL, BLK_PERM_WRITE, BLK_PERM_WRITE_UNCHANGED, DEFAULT_PERM_PASSTHROUGH,
    DEFAULT_PERM_UNCHANGED, perm_names,
};

/// `BDRV_SECTOR_SIZE`. Lengths are reported in whole sectors, as `bdrv_getlength()` does.
pub(crate) const BDRV_SECTOR_SIZE: u64 = 512;

/// The largest single write the zero fallback makes, like `MAX_WRITE_ZEROES_BOUNCE_BUFFER`.
const MAX_ZERO_BOUNCE: usize = 32768 * BDRV_SECTOR_SIZE as usize;

/// `ENOMEDIUM`, which QEMU defines as `ENODEV` on hosts that lack it.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub(crate) const ENOMEDIUM: i32 = libc::ENOMEDIUM;
/// `ENOMEDIUM`, which QEMU defines as `ENODEV` on hosts that lack it.
#[cfg(not(any(target_os = "linux", target_os = "android")))]
pub(crate) const ENOMEDIUM: i32 = libc::ENODEV;

/// A host error number as an `io::Error`, the way QEMU returns `-errno`.
pub(crate) fn errno(e: i32) -> io::Error {
    io::Error::from_raw_os_error(e)
}

pub(crate) fn is_enotsup(e: &io::Error) -> bool {
    matches!(e.raw_os_error(), Some(n) if n == libc::ENOTSUP || n == libc::EOPNOTSUPP)
}

/// The generic open flags of a node, the parts of `bs->open_flags` and friends that the request
/// layer looks at.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct NodeFlags {
    /// Not `BDRV_O_RDWR`.
    pub read_only: bool,
    /// `BDRV_O_NOCACHE`, `cache.direct`.
    pub direct: bool,
    /// `BDRV_O_NO_FLUSH`, `cache.no-flush`.
    pub no_flush: bool,
    /// `BDRV_O_UNMAP`, `discard=unmap`.
    pub unmap: bool,
    /// `force-share`.
    pub force_share: bool,
    /// `detect-zeroes`.
    pub detect_zeroes: BlockdevDetectZeroesOptions,
}

/// The driver callbacks, the synchronous subset of `BlockDriver` that the drivers so far need.
/// A method a driver does not implement behaves like a NULL callback in QEMU.
pub(crate) trait Driver: Send + Sync {
    /// `.bdrv_co_preadv`. The range is inside the node, the generic layer checked that.
    fn pread(&self, bs: &Node, offset: u64, buf: &mut [u8]) -> io::Result<()>;

    /// `.bdrv_co_pwritev`.
    fn pwrite(&self, bs: &Node, offset: u64, buf: &[u8]) -> io::Result<()>;

    /// `.bdrv_co_pwrite_zeroes`. `ENOTSUP` makes the generic layer write a buffer of zeroes.
    fn pwrite_zeroes(&self, bs: &Node, offset: u64, bytes: u64, may_unmap: bool) -> io::Result<()> {
        let _ = (bs, offset, bytes, may_unmap);
        Err(errno(libc::ENOTSUP))
    }

    /// `.bdrv_co_pdiscard`. `ENOTSUP` is not an error, discard is only advice.
    fn pdiscard(&self, bs: &Node, offset: u64, bytes: u64) -> io::Result<()> {
        let _ = (bs, offset, bytes);
        Err(errno(libc::ENOTSUP))
    }

    /// `.bdrv_co_flush_to_disk`.
    fn flush_to_disk(&self, bs: &Node) -> io::Result<()> {
        let _ = bs;
        Ok(())
    }

    /// The host file behind a protocol driver, for `bs->filename`.
    fn filename(&self) -> Option<String> {
        None
    }

    /// `.bdrv_co_getlength`, in bytes and not rounded.
    fn getlength(&self, bs: &Node) -> io::Result<u64>;

    /// `.bdrv_co_truncate`.
    fn truncate(&self, bs: &Node, len: u64) -> Result<()> {
        let _ = (bs, len);
        Err(Error::generic("Image format driver does not support resize"))
    }

    /// `.bdrv_child_perm` for child number `index`. The default is `bdrv_filter_default_perms()`.
    fn child_perm(&self, index: usize, perm: u64, shared: u64) -> (u64, u64) {
        let _ = index;
        (
            perm & DEFAULT_PERM_PASSTHROUGH,
            (shared & DEFAULT_PERM_PASSTHROUGH) | DEFAULT_PERM_UNCHANGED,
        )
    }

    /// `.bdrv_check_perm` followed by `.bdrv_set_perm`: take the cumulative permissions of the
    /// parents, or fail and leave the old ones in place.
    fn set_perm(&self, perm: u64, shared: u64) -> Result<()> {
        let _ = (perm, shared);
        Ok(())
    }
}

/// One parent of a node and what it holds, a `BdrvChild` seen from the child.
#[derive(Clone, Debug)]
pub(crate) struct Parent {
    pub id: u64,
    /// `bdrv_child_user_desc()`: `node 'x'`, `block device 'x'` or `an unnamed block device`.
    pub desc: String,
    /// The name of the edge: `file`, `image` or `root`.
    pub child_name: &'static str,
    pub perm: u64,
    pub shared: u64,
}

static NEXT_EDGE: AtomicU64 = AtomicU64::new(1);

/// A fresh identifier for a parent edge.
pub(crate) fn new_edge_id() -> u64 {
    NEXT_EDGE.fetch_add(1, Ordering::Relaxed)
}

/// An edge from a node to one of its children.
pub(crate) struct Child {
    pub name: &'static str,
    pub node: Arc<Node>,
    edge: u64,
}

/// A `BlockDriverState`.
pub(crate) struct Node {
    pub name: String,
    pub driver_name: &'static str,
    pub driver: Box<dyn Driver>,
    pub flags: NodeFlags,
    pub children: Vec<Child>,
    /// Newest first, like `bs->parents`, so error messages name parents in QEMU's order.
    parents: Mutex<Vec<Parent>>,
}

impl std::fmt::Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("name", &self.name)
            .field("driver", &self.driver_name)
            .field("flags", &self.flags)
            .finish_non_exhaustive()
    }
}

impl Node {
    /// Makes a node and attaches it to its children with the permissions it needs while it has
    /// no parents of its own.
    pub(crate) fn new(
        name: String,
        driver_name: &'static str,
        driver: Box<dyn Driver>,
        flags: NodeFlags,
        children: Vec<(&'static str, Arc<Node>)>,
    ) -> Result<Arc<Node>> {
        let children = children
            .into_iter()
            .map(|(name, node)| Child { name, node, edge: new_edge_id() })
            .collect();
        let node = Arc::new(Node {
            name,
            driver_name,
            driver,
            flags,
            children,
            parents: Mutex::new(Vec::new()),
        });
        // On failure dropping `node` takes back whatever edges were attached.
        node.refresh_children(0, BLK_PERM_ALL)?;
        Ok(node)
    }

    /// The primary child, `bs->file` for the drivers here.
    pub(crate) fn file(&self) -> &Arc<Node> {
        &self.children[0].node
    }

    /// `bs->filename`: the host file of a protocol node, or of the primary child of a format
    /// or filter node.
    pub(crate) fn filename(&self) -> Option<String> {
        self.driver.filename().or_else(|| self.children.first().and_then(|c| c.node.filename()))
    }

    /// How many parents hold this node, nodes and block backends alike.
    pub(crate) fn parent_count(&self) -> usize {
        self.parents.lock().unwrap().len()
    }

    /// Adds, replaces or (with `None`) removes the parent edge `id`, and updates the permissions
    /// down the graph. On failure everything is as it was, as with QEMU's permission
    /// transaction.
    pub(crate) fn update_parent(&self, id: u64, new: Option<Parent>) -> Result<()> {
        let mut parents = self.parents.lock().unwrap();
        let old = parents.clone();
        match new {
            Some(p) => match parents.iter_mut().find(|q| q.id == id) {
                Some(slot) => *slot = p,
                None => parents.insert(0, p),
            },
            None => parents.retain(|q| q.id != id),
        }
        if let Err(e) = self.refresh(&parents) {
            *parents = old;
            // Going back only ever gives up permissions, which cannot conflict.
            let _ = self.refresh(&parents);
            return Err(e);
        }
        Ok(())
    }

    /// `bdrv_node_refresh_perm()` plus the conflict check of `bdrv_parent_perms_conflict()`.
    fn refresh(&self, parents: &[Parent]) -> Result<()> {
        for a in parents {
            for b in parents {
                if a.id == b.id || b.perm & a.shared == b.perm {
                    continue;
                }
                let n = &self.name;
                return Err(Error::generic(format!(
                    "Permission conflict on node '{n}': permissions '{}' are both required by \
                     {} (uses node '{n}' as '{}' child) and unshared by {} (uses node '{n}' as \
                     '{}' child).",
                    perm_names(b.perm & !a.shared),
                    b.desc,
                    b.child_name,
                    a.desc,
                    a.child_name
                )));
            }
        }
        let (perm, shared) = cumulative(parents);
        if perm & (BLK_PERM_WRITE | BLK_PERM_WRITE_UNCHANGED) != 0 && self.flags.read_only {
            return Err(Error::generic("Block node is read-only"));
        }
        self.driver.set_perm(perm, shared)?;
        self.refresh_children(perm, shared)
    }

    fn refresh_children(&self, perm: u64, shared: u64) -> Result<()> {
        for (i, c) in self.children.iter().enumerate() {
            let (cperm, mut cshared) = self.driver.child_perm(i, perm, shared);
            if c.node.flags.force_share {
                cshared = BLK_PERM_ALL;
            }
            c.node.update_parent(
                c.edge,
                Some(Parent {
                    id: c.edge,
                    desc: format!("node '{}'", self.name),
                    child_name: c.name,
                    perm: cperm,
                    shared: cshared,
                }),
            )?;
        }
        Ok(())
    }

    /// `bdrv_co_getlength()`: the driver's length rounded up to whole sectors.
    pub(crate) fn getlength(&self) -> io::Result<u64> {
        let len = self.driver.getlength(self)?;
        len.checked_next_multiple_of(BDRV_SECTOR_SIZE).ok_or_else(|| errno(libc::EFBIG))
    }

    /// `bdrv_co_preadv()`.
    pub(crate) fn pread(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        check_request(offset, buf.len() as u64)?;
        if buf.is_empty() {
            return Ok(());
        }
        self.driver.pread(self, offset, buf)
    }

    /// `bdrv_co_pwritev()`, with `detect-zeroes` turning an all zero write into a zero write.
    pub(crate) fn pwrite(&self, offset: u64, buf: &[u8]) -> io::Result<()> {
        check_request(offset, buf.len() as u64)?;
        if self.flags.read_only {
            return Err(errno(libc::EPERM));
        }
        if buf.is_empty() {
            return Ok(());
        }
        if self.flags.detect_zeroes != BlockdevDetectZeroesOptions::Off
            && buf.iter().all(|&b| b == 0)
        {
            let unmap = self.flags.detect_zeroes == BlockdevDetectZeroesOptions::Unmap;
            return self.do_pwrite_zeroes(offset, buf.len() as u64, unmap);
        }
        self.driver.pwrite(self, offset, buf)
    }

    /// `bdrv_co_pwrite_zeroes()`. `may_unmap` is `BDRV_REQ_MAY_UNMAP`, dropped when the node
    /// was not opened with `discard=unmap`.
    pub(crate) fn pwrite_zeroes(&self, offset: u64, bytes: u64, may_unmap: bool) -> io::Result<()> {
        check_request(offset, bytes)?;
        if self.flags.read_only {
            return Err(errno(libc::EPERM));
        }
        if bytes == 0 {
            return Ok(());
        }
        self.do_pwrite_zeroes(offset, bytes, may_unmap)
    }

    /// `bdrv_co_do_pwrite_zeroes()`: the driver's way, or a bounce buffer of zeroes.
    fn do_pwrite_zeroes(&self, offset: u64, bytes: u64, may_unmap: bool) -> io::Result<()> {
        let may_unmap = may_unmap && self.flags.unmap;
        match self.driver.pwrite_zeroes(self, offset, bytes, may_unmap) {
            Err(e) if is_enotsup(&e) => {}
            r => return r,
        }
        let chunk = bytes.min(MAX_ZERO_BOUNCE as u64) as usize;
        let zeroes = vec![0u8; chunk];
        let mut done = 0;
        while done < bytes {
            let n = (bytes - done).min(chunk as u64) as usize;
            self.driver.pwrite(self, offset + done, &zeroes[..n])?;
            done += n as u64;
        }
        Ok(())
    }

    /// `bdrv_co_pdiscard()`: nothing unless the node was opened with `discard=unmap`, and a
    /// driver that cannot discard is not an error.
    pub(crate) fn pdiscard(&self, offset: u64, bytes: u64) -> io::Result<()> {
        check_request(offset, bytes)?;
        if !self.flags.unmap || bytes == 0 {
            return Ok(());
        }
        if self.flags.read_only {
            return Err(errno(libc::EPERM));
        }
        match self.driver.pdiscard(self, offset, bytes) {
            Err(e) if is_enotsup(&e) => Ok(()),
            r => r,
        }
    }

    /// `bdrv_co_flush()`: this node unless it has `cache.no-flush`, then its children.
    pub(crate) fn flush(&self) -> io::Result<()> {
        if self.flags.read_only {
            return Ok(());
        }
        let mut ret = Ok(());
        if !self.flags.no_flush {
            ret = self.driver.flush_to_disk(self);
        }
        for c in &self.children {
            let r = c.node.flush();
            if ret.is_ok() {
                ret = r;
            }
        }
        ret
    }

    /// `bdrv_co_truncate()` with `PREALLOC_MODE_OFF`.
    pub(crate) fn truncate(&self, len: u64) -> Result<()> {
        if len > i64::MAX as u64 {
            return Err(Error::generic("Image size cannot be negative"));
        }
        if self.flags.read_only {
            return Err(Error::generic("Image is read-only"));
        }
        self.driver.truncate(self, len)
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        for c in &self.children {
            let _ = c.node.update_parent(c.edge, None);
        }
    }
}

fn cumulative(parents: &[Parent]) -> (u64, u64) {
    parents.iter().fold((0, BLK_PERM_ALL), |(p, s), q| (p | q.perm, s & q.shared))
}

/// `bdrv_check_request()`: offsets and lengths fit in an `int64_t`.
fn check_request(offset: u64, bytes: u64) -> io::Result<()> {
    if offset > i64::MAX as u64 || bytes > i64::MAX as u64 - offset {
        return Err(errno(libc::EIO));
    }
    Ok(())
}
