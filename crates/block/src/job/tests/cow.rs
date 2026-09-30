// SPDX-License-Identifier: GPL-2.0-or-later

//! A copy-on-write format in memory for the job tests, `bdrv_test` of test-bdrv-drain.c
//! grown into something the stream, commit, mirror and backup tests can check data with: it
//! keeps track of which 512-byte sectors it has and reads the others from its backing child.

use std::io;
use std::sync::atomic::{AtomicI32, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use ruvm_base::Result;
use ruvm_qapi::types::BlockdevOptionsU;

use crate::drivers::{DriverDef, OpenArgs};
use crate::node::{BDRV_BLOCK_DATA, BlockStatus, Driver, Node, NodeFlags, NodeMeta, NodeSpec};

pub(crate) const SECTOR: u64 = 512;

fn no_open(_: &mut OpenArgs<'_>, _: BlockdevOptionsU) -> Result<Box<dyn Driver>> {
    unreachable!("test drivers are not opened from options")
}

/// `bdrv_test`, with backing files.
static BDRV_TEST_COW: DriverDef = DriverDef::format("testcow", no_open).with_backing();

/// The state of a test image.
#[derive(Default)]
pub(crate) struct CowState {
    pub data: Mutex<Vec<u8>>,
    pub allocated: Mutex<Vec<bool>>,
    /// `drain_count`.
    pub drain_count: AtomicI32,
    pub flushes: AtomicU32,
    /// Fail writes (positive errno) while not 0.
    pub fail_write: AtomicI32,
    /// Fail reads (positive errno) while not 0.
    pub fail_read: AtomicI32,
}

struct CowDriver(Arc<CowState>);

impl Driver for CowDriver {
    fn pread(&self, bs: &Node, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let e = self.0.fail_read.load(Ordering::SeqCst);
        if e != 0 {
            return Err(io::Error::from_raw_os_error(e));
        }
        let backing = bs.backing().map(|c| c.node);
        let mut off = 0;
        while off < buf.len() {
            let pos = offset + off as u64;
            let n = (SECTOR - pos % SECTOR).min((buf.len() - off) as u64) as usize;
            let chunk = &mut buf[off..off + n];
            let alloc = self.0.allocated.lock().unwrap()[(pos / SECTOR) as usize];
            if alloc {
                chunk.copy_from_slice(&self.0.data.lock().unwrap()[pos as usize..pos as usize + n]);
            } else if let Some(b) = &backing {
                let blen = b.getlength()?;
                if pos >= blen {
                    chunk.fill(0);
                } else {
                    let m = (n as u64).min(blen - pos) as usize;
                    b.pread(pos, &mut chunk[..m])?;
                    chunk[m..].fill(0);
                }
            } else {
                chunk.fill(0);
            }
            off += n;
        }
        Ok(())
    }

    fn pwrite(&self, _: &Node, offset: u64, buf: &[u8]) -> io::Result<()> {
        let e = self.0.fail_write.load(Ordering::SeqCst);
        if e != 0 {
            return Err(io::Error::from_raw_os_error(e));
        }
        self.0.data.lock().unwrap()[offset as usize..offset as usize + buf.len()]
            .copy_from_slice(buf);
        let first = offset / SECTOR;
        let last = (offset + buf.len() as u64).div_ceil(SECTOR);
        let mut a = self.0.allocated.lock().unwrap();
        for i in first..last {
            a[i as usize] = true;
        }
        Ok(())
    }

    fn getlength(&self, _: &Node) -> io::Result<u64> {
        Ok(self.0.data.lock().unwrap().len() as u64)
    }

    fn truncate(&self, _: &Node, len: u64) -> Result<()> {
        self.0.data.lock().unwrap().resize(len as usize, 0);
        self.0.allocated.lock().unwrap().resize(len.div_ceil(SECTOR) as usize, false);
        Ok(())
    }

    fn flush_to_disk(&self, _: &Node) -> io::Result<()> {
        self.0.flushes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn block_status(
        &self,
        _: &Node,
        _: u32,
        offset: u64,
        bytes: u64,
    ) -> Option<io::Result<BlockStatus>> {
        let a = self.0.allocated.lock().unwrap();
        let first = (offset / SECTOR) as usize;
        let alloc = a[first];
        let mut pnum = 0;
        while pnum < bytes
            && first + ((pnum / SECTOR) as usize) < a.len()
            && a[first + (pnum / SECTOR) as usize] == alloc
        {
            pnum += SECTOR;
        }
        let ret = if alloc { BDRV_BLOCK_DATA } else { 0 };
        Some(Ok(BlockStatus { ret, pnum: pnum.min(bytes), map: 0, file: None }))
    }

    fn change_backing_file(
        &self,
        _: &Node,
        _: Option<&str>,
        _: Option<&str>,
    ) -> Option<io::Result<()>> {
        Some(Ok(()))
    }

    fn make_empty(&self, _: &Node) -> Option<io::Result<()>> {
        self.0.allocated.lock().unwrap().fill(false);
        Some(Ok(()))
    }

    fn drain_begin(&self, _: &Node) {
        self.0.drain_count.fetch_add(1, Ordering::SeqCst);
    }

    fn drain_end(&self, _: &Node) {
        self.0.drain_count.fetch_sub(1, Ordering::SeqCst);
    }
}

/// `bdrv_new_open_driver(&bdrv_test, name, BDRV_O_RDWR)`: an empty image of `size` bytes.
pub(crate) fn cow_node(name: &str, size: u64) -> (Arc<Node>, Arc<CowState>) {
    let s = Arc::new(CowState {
        data: Mutex::new(vec![0; size as usize]),
        allocated: Mutex::new(vec![false; size.div_ceil(SECTOR) as usize]),
        ..CowState::default()
    });
    let node = Node::build(NodeSpec {
        name: name.to_string(),
        driver_name: "testcow",
        driver: Box::new(CowDriver(s.clone())),
        def: Some(&BDRV_TEST_COW),
        flags: NodeFlags::default(),
        meta: NodeMeta::default(),
        children: Vec::new(),
    })
    .unwrap();
    (node, s)
}

impl CowState {
    /// Whether the image has the sector at `offset` itself.
    pub(crate) fn has(&self, offset: u64) -> bool {
        self.allocated.lock().unwrap()[(offset / SECTOR) as usize]
    }
}
