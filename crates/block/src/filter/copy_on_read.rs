// SPDX-License-Identifier: GPL-2.0-or-later

//! `copy-on-read` from block/copy-on-read.c: a filter whose reads copy the data they find in
//! the backing chain of its child into the child, so that the child ends up holding it.
//!
//! Without `bottom` every read goes to the child with `BDRV_REQ_COPY_ON_READ`. With `bottom`,
//! only data allocated in the backing chain of the child down to and including the bottom
//! node is copied; data below it is read but left where it is.
//!
//! Differences from QEMU:
//!
//! - There are no frozen backing chains, so the chain between the child and the bottom node
//!   is not frozen and can change under the filter.
//! - A bottom node that is not in the backing chain of the child makes QEMU's
//!   `bdrv_freeze_backing_chain()` loop forever. Here it is accepted and the whole chain of
//!   the child counts as above the bottom.
//! - "Bottom node '%s' not opened" cannot happen: a node here always has its driver.
//! - The generic read path has no per-driver read flags, so `BDRV_REQ_PREFETCH` requests
//!   never reach the filter; the generic layer finishes them for the filter itself.

use std::io;
use std::sync::Arc;

use ruvm_base::{Error, Result};
use ruvm_qapi::types::BlockdevOptionsU;

use crate::drivers::{DriverDef, OpenArgs};
use crate::node::{
    BDRV_CHILD_FILTERED, BDRV_CHILD_PRIMARY, BDRV_REQ_COPY_ON_READ, BDRV_REQ_FUA,
    BDRV_REQ_MAY_UNMAP, BDRV_REQ_NO_FALLBACK, BDRV_REQ_PREFETCH, BDRV_REQ_WRITE_COMPRESSED,
    BDRV_REQ_WRITE_UNCHANGED, Driver, Node, filter_default_perms,
};
use crate::perm::{BLK_PERM_WRITE_UNCHANGED, PermCtx};

/// `bdrv_copy_on_read`.
pub(crate) static COPY_ON_READ: DriverDef = DriverDef::filter("copy-on-read", cor_open);

/// `BDRVStateCOR` plus the flags `cor_open()` puts in `bs`.
struct CorDriver {
    /// `bottom_bs`.
    bottom: Option<Arc<Node>>,
    /// `bs->supported_write_flags`.
    write_flags: u32,
    /// `bs->supported_zero_flags`.
    zero_flags: u32,
}

/// `cor_open()`.
fn cor_open(args: &mut OpenArgs<'_>, opts: BlockdevOptionsU) -> Result<Box<dyn Driver>> {
    let BlockdevOptionsU::CopyOnRead(o) = opts else {
        unreachable!("the copy-on-read driver gets copy-on-read options");
    };
    let file = args.open_child(*o.file, "file", BDRV_CHILD_FILTERED | BDRV_CHILD_PRIMARY)?;

    let write_flags =
        BDRV_REQ_WRITE_UNCHANGED | (BDRV_REQ_FUA & file.driver.supported_write_flags());
    let zero_flags = BDRV_REQ_WRITE_UNCHANGED
        | ((BDRV_REQ_FUA | BDRV_REQ_MAY_UNMAP | BDRV_REQ_NO_FALLBACK)
            & file.driver.supported_zero_flags());

    let mut bottom = None;
    if let Some(name) = &o.bottom {
        let found = args
            .pending
            .nodes
            .iter()
            .find(|n| n.name == *name)
            .cloned()
            .or_else(|| args.graph.find_node(name));
        let Some(b) = found else {
            return Err(Error::generic(format!("Bottom node '{name}' not found")));
        };
        if b.is_filter() {
            return Err(Error::generic(format!("Bottom node '{name}' is a filter")));
        }
        bottom = Some(b);
    }

    Ok(Box::new(CorDriver { bottom, write_flags, zero_flags }))
}

impl CorDriver {
    /// `cor_co_preadv_part()` with a bottom node: copy only what is allocated above it.
    fn preadv_bottom(
        &self,
        bs: &Node,
        bottom: &Arc<Node>,
        mut offset: u64,
        buf: &mut [u8],
        flags: u32,
    ) -> io::Result<()> {
        let file = bs.file();
        let mut done = 0usize;
        let mut bytes = buf.len() as u64;
        while bytes > 0 {
            let mut local_flags = flags;
            // In case of failure, try to copy-on-read anyway.
            let (allocated, mut n) = file.is_allocated(offset, bytes).unwrap_or((false, bytes));
            if !allocated {
                let above = match file.backing_chain_next() {
                    Some(next) => next.is_allocated_above(Some(bottom), true, offset, n),
                    None => Ok((0, n)),
                };
                match above {
                    Ok((depth, pnum)) => {
                        n = pnum;
                        if depth > 0 {
                            local_flags |= BDRV_REQ_COPY_ON_READ;
                        }
                    }
                    Err(_) => local_flags |= BDRV_REQ_COPY_ON_READ,
                }
                // Finish earlier if the end of a backing file has been reached.
                if n == 0 {
                    break;
                }
            }

            // Skip if neither read nor write are needed.
            if local_flags & (BDRV_REQ_PREFETCH | BDRV_REQ_COPY_ON_READ) != BDRV_REQ_PREFETCH {
                file.preadv_flags(offset, &mut buf[done..done + n as usize], local_flags)?;
            }

            offset += n;
            done += n as usize;
            bytes -= n;
        }
        Ok(())
    }
}

impl Driver for CorDriver {
    fn pread(&self, bs: &Node, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        match &self.bottom {
            None => bs.file().preadv_flags(offset, buf, BDRV_REQ_COPY_ON_READ),
            Some(b) => self.preadv_bottom(bs, b, offset, buf, 0),
        }
    }

    fn pwrite(&self, bs: &Node, offset: u64, buf: &[u8]) -> io::Result<()> {
        self.pwrite_flags(bs, offset, buf, 0)
    }

    fn pwrite_flags(&self, bs: &Node, offset: u64, buf: &[u8], flags: u32) -> io::Result<()> {
        bs.file().pwrite_flags(offset, buf, flags)
    }

    fn supported_write_flags(&self) -> u32 {
        self.write_flags
    }

    fn pwrite_zeroes(&self, bs: &Node, offset: u64, bytes: u64, may_unmap: bool) -> io::Result<()> {
        self.pwrite_zeroes_flags(bs, offset, bytes, if may_unmap { BDRV_REQ_MAY_UNMAP } else { 0 })
    }

    fn pwrite_zeroes_flags(
        &self,
        bs: &Node,
        offset: u64,
        bytes: u64,
        flags: u32,
    ) -> io::Result<()> {
        bs.file().pwrite_zeroes_flags(offset, bytes, flags)
    }

    fn supported_zero_flags(&self) -> u32 {
        self.zero_flags
    }

    fn pdiscard(&self, bs: &Node, offset: u64, bytes: u64) -> io::Result<()> {
        bs.file().pdiscard(offset, bytes)
    }

    fn pwrite_compressed(&self, bs: &Node, offset: u64, buf: &[u8]) -> Option<io::Result<()>> {
        Some(bs.file().pwrite_flags(offset, buf, BDRV_REQ_WRITE_COMPRESSED))
    }

    fn can_compress(&self) -> bool {
        true
    }

    fn getlength(&self, bs: &Node) -> io::Result<u64> {
        bs.file().getlength()
    }

    fn has_truncate(&self) -> bool {
        false
    }

    fn child_perm_for(&self, ctx: &PermCtx<'_>, perm: u64, shared: u64) -> (u64, u64) {
        let (mut p, s) = filter_default_perms(perm, shared);
        // We must not request write permissions for an inactive node, the child cannot
        // provide it.
        if !ctx.inactive {
            p |= BLK_PERM_WRITE_UNCHANGED;
        }
        (p, s)
    }

    fn eject(&self, bs: &Node, eject_flag: bool) {
        let file = bs.file();
        file.driver.eject(&file, eject_flag);
    }

    fn lock_medium(&self, bs: &Node, locked: bool) {
        let file = bs.file();
        file.driver.lock_medium(&file, locked);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::node::{
        BDRV_BLOCK_DATA, BDRV_BLOCK_OFFSET_VALID, BlockStatus, NodeFlags, NodeMeta, NodeSpec,
    };

    /// An image in memory with a per-sector allocation map. Unallocated sectors read from
    /// the `backing` child, or as zeroes without one.
    #[derive(Default)]
    pub(crate) struct Mem {
        pub data: Mutex<Vec<u8>>,
        pub alloc: Mutex<Vec<bool>>,
        /// Offsets and lengths of the compressed writes.
        pub compressed: Mutex<Vec<(u64, usize)>>,
        /// Whether it takes compressed writes.
        pub compress: bool,
        /// Zero writes seen, with their flags.
        pub zeroes: Mutex<Vec<(u64, u64, u32)>>,
    }

    impl Mem {
        pub(crate) fn with_data(data: Vec<u8>, allocated: bool) -> Self {
            let sectors = data.len().div_ceil(512);
            Mem {
                data: Mutex::new(data),
                alloc: Mutex::new(vec![allocated; sectors]),
                ..Mem::default()
            }
        }

        fn store(&self, offset: u64, buf: Option<&[u8]>, bytes: u64) {
            let mut d = self.data.lock().unwrap();
            let end = (offset + bytes) as usize;
            if d.len() < end {
                d.resize(end, 0);
            }
            match buf {
                Some(b) => d[offset as usize..end].copy_from_slice(b),
                None => d[offset as usize..end].fill(0),
            }
            let mut a = self.alloc.lock().unwrap();
            let sectors = d.len().div_ceil(512);
            if a.len() < sectors {
                a.resize(sectors, false);
            }
            for s in (offset / 512) as usize..end.div_ceil(512) {
                a[s] = true;
            }
        }
    }

    impl Driver for Mem {
        fn pread(&self, bs: &Node, offset: u64, buf: &mut [u8]) -> io::Result<()> {
            let backing = bs.backing().map(|c| c.node);
            for (i, chunk) in buf.chunks_mut(512).enumerate() {
                let off = offset + i as u64 * 512;
                let allocated = self.alloc.lock().unwrap().get((off / 512) as usize) == Some(&true);
                if allocated {
                    let d = self.data.lock().unwrap();
                    let o = off as usize;
                    chunk.copy_from_slice(&d[o..o + chunk.len()]);
                } else if let Some(b) = &backing {
                    b.pread(off, chunk)?;
                } else {
                    chunk.fill(0);
                }
            }
            Ok(())
        }

        fn pwrite(&self, _bs: &Node, offset: u64, buf: &[u8]) -> io::Result<()> {
            self.store(offset, Some(buf), buf.len() as u64);
            Ok(())
        }

        fn supported_write_flags(&self) -> u32 {
            BDRV_REQ_FUA
        }

        fn pwrite_zeroes_flags(
            &self,
            _bs: &Node,
            offset: u64,
            bytes: u64,
            flags: u32,
        ) -> io::Result<()> {
            self.zeroes.lock().unwrap().push((offset, bytes, flags));
            self.store(offset, None, bytes);
            Ok(())
        }

        fn supported_zero_flags(&self) -> u32 {
            BDRV_REQ_MAY_UNMAP | BDRV_REQ_NO_FALLBACK
        }

        fn pwrite_compressed(&self, _bs: &Node, offset: u64, buf: &[u8]) -> Option<io::Result<()>> {
            if !self.compress {
                return None;
            }
            self.compressed.lock().unwrap().push((offset, buf.len()));
            self.store(offset, Some(buf), buf.len() as u64);
            Some(Ok(()))
        }

        fn can_compress(&self) -> bool {
            self.compress
        }

        fn getlength(&self, _bs: &Node) -> io::Result<u64> {
            Ok(self.data.lock().unwrap().len() as u64)
        }

        fn truncate(&self, _bs: &Node, len: u64) -> Result<()> {
            self.data.lock().unwrap().resize(len as usize, 0);
            self.alloc.lock().unwrap().resize((len as usize).div_ceil(512), false);
            Ok(())
        }

        fn block_status(
            &self,
            _bs: &Node,
            _want: u32,
            offset: u64,
            bytes: u64,
        ) -> Option<io::Result<BlockStatus>> {
            let a = self.alloc.lock().unwrap();
            let first = (offset / 512) as usize;
            let state = a.get(first).copied().unwrap_or(false);
            let mut n = 0u64;
            while n < bytes && a.get(first + (n / 512) as usize).copied().unwrap_or(false) == state
            {
                n += 512;
            }
            let ret = if state { BDRV_BLOCK_DATA | BDRV_BLOCK_OFFSET_VALID } else { 0 };
            Some(Ok(BlockStatus { ret, pnum: n.min(bytes).max(512), map: offset, file: None }))
        }

        fn as_any(&self) -> Option<&dyn std::any::Any> {
            Some(self)
        }
    }

    /// A format node over `Mem` whose `backing` child, if any, supports backing files.
    pub(crate) fn mem_node(name: &str, mem: Mem, backing: Option<Arc<Node>>) -> Arc<Node> {
        fn open(_: &mut OpenArgs<'_>, _: BlockdevOptionsU) -> Result<Box<dyn Driver>> {
            unreachable!("test nodes are built directly")
        }
        static FMT: DriverDef = DriverDef::format("test-mem", open).with_backing();
        let children = backing
            .into_iter()
            .map(|b| ("backing".to_string(), b, crate::node::BDRV_CHILD_COW))
            .collect();
        Node::build(NodeSpec {
            name: name.to_string(),
            driver_name: "test-mem",
            driver: Box::new(mem),
            def: Some(&FMT),
            flags: NodeFlags::default(),
            meta: NodeMeta::default(),
            children,
        })
        .unwrap()
    }

    pub(crate) fn mem(n: &Node) -> &Mem {
        n.driver.as_any().unwrap().downcast_ref::<Mem>().unwrap()
    }

    pub(crate) fn allocated(n: &Node, sector: usize) -> bool {
        mem(n).alloc.lock().unwrap().get(sector) == Some(&true)
    }

    fn cor_node(file: Arc<Node>, bottom: Option<Arc<Node>>) -> Arc<Node> {
        let write_flags =
            BDRV_REQ_WRITE_UNCHANGED | (BDRV_REQ_FUA & file.driver.supported_write_flags());
        let driver = CorDriver { bottom, write_flags, zero_flags: BDRV_REQ_WRITE_UNCHANGED };
        Node::build(NodeSpec {
            name: "cor".into(),
            driver_name: "copy-on-read",
            driver: Box::new(driver),
            def: Some(&COPY_ON_READ),
            flags: NodeFlags::default(),
            meta: NodeMeta::default(),
            children: vec![("file".into(), file, BDRV_CHILD_FILTERED | BDRV_CHILD_PRIMARY)],
        })
        .unwrap()
    }

    #[test]
    fn populates_top() {
        let base = mem_node("base", Mem::with_data(vec![0xab; 4096], true), None);
        let top = mem_node("top", Mem::with_data(vec![0; 4096], false), Some(base));
        let cor = cor_node(top.clone(), None);
        assert!(!allocated(&top, 2));

        let mut buf = vec![0u8; 1024];
        cor.pread(1024, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0xab));
        assert!(allocated(&top, 2) && allocated(&top, 3));
        assert!(!allocated(&top, 0) && !allocated(&top, 7));
        assert_eq!(mem(&top).data.lock().unwrap()[1024], 0xab);

        // Writes go to the child unchanged.
        cor.pwrite(0, &[1u8; 512]).unwrap();
        assert!(allocated(&top, 0));
        assert_eq!(mem(&top).data.lock().unwrap()[0], 1);
        assert_eq!(cor.getlength().unwrap(), 4096);
    }

    #[test]
    fn bottom_limits_copying() {
        // base (allocated) <- mid (sector 1 allocated) <- top (empty). With mid as the bottom
        // node only what mid holds is copied up; base's data is read but not copied.
        let base = mem_node("base", Mem::with_data(vec![0xbb; 2048], true), None);
        let mid_mem = Mem::with_data(vec![0; 2048], false);
        mid_mem.data.lock().unwrap()[512..1024].fill(0xcc);
        mid_mem.alloc.lock().unwrap()[1] = true;
        let mid = mem_node("mid", mid_mem, Some(base));
        let top = mem_node("top", Mem::with_data(vec![0; 2048], false), Some(mid.clone()));
        let cor = cor_node(top.clone(), Some(mid));

        let mut buf = vec![0u8; 2048];
        cor.pread(0, &mut buf).unwrap();
        assert!(buf[..512].iter().all(|&b| b == 0xbb));
        assert!(buf[512..1024].iter().all(|&b| b == 0xcc));
        assert!(buf[1024..].iter().all(|&b| b == 0xbb));
        assert!(!allocated(&top, 0));
        assert!(allocated(&top, 1));
        assert!(!allocated(&top, 2) && !allocated(&top, 3));
    }

    #[test]
    fn open_errors() {
        use ruvm_qapi::QDict;
        let g = crate::graph::BlockGraph::new();
        let mut o = QDict::new();
        o.put("driver", "copy-on-read");
        o.put("file.driver", "null-co");
        o.put("bottom", "nope");
        let e = g.open_image(None, o).unwrap_err();
        assert_eq!(e.message(), "Bottom node 'nope' not found");

        let mut o = QDict::new();
        o.put("driver", "null-co");
        o.put("node-name", "n0");
        g.open_image(None, o.clone()).unwrap();
        let mut o = QDict::new();
        o.put("driver", "copy-on-read");
        o.put("file", "n0");
        o.put("node-name", "cor0");
        assert_eq!(g.open_image(None, o).unwrap(), "cor0");

        let mut o = QDict::new();
        o.put("driver", "copy-on-read");
        o.put("file", "n0");
        o.put("bottom", "cor0");
        let e = g.open_image(None, o).unwrap_err();
        assert_eq!(e.message(), "Bottom node 'cor0' is a filter");
    }
}
