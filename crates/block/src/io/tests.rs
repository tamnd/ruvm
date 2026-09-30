// SPDX-License-Identifier: GPL-2.0-or-later

//! Tests of the request layer: a port of tests/unit/test-write-threshold.c, and checks of
//! alignment padding with read-modify-write, `max_transfer` splitting, serialising requests,
//! copy-on-read and block status through a backing chain.
//!
//! The tests run against [`MemDriver`], an image in memory that logs every request the
//! driver sees and keeps track of which sectors it allocated, reading the others from its
//! backing child like a format driver would.

use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ruvm_base::Result;

use super::ReqType;
use crate::node::{
    BDRV_BLOCK_ALLOCATED, BDRV_BLOCK_DATA, BlockLimits, BlockStatus, Driver, Node, NodeFlags,
};

const SECTOR: u64 = 512;

/// One request as the driver saw it.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Op {
    Read(u64, u64),
    Write(u64, u64),
}

#[derive(Default)]
struct MemState {
    data: Vec<u8>,
    allocated: Vec<bool>,
    log: Vec<Op>,
}

/// An image in memory, sector by sector.
struct MemDriver {
    state: Arc<Mutex<MemState>>,
    align: u32,
    max_transfer: u32,
}

impl MemDriver {
    fn node(
        name: &str,
        size: u64,
        align: u32,
        max_transfer: u32,
    ) -> (Arc<Node>, Arc<Mutex<MemState>>) {
        Self::node_with_backing(name, size, align, max_transfer, None)
    }

    fn node_with_backing(
        name: &str,
        size: u64,
        align: u32,
        max_transfer: u32,
        backing: Option<Arc<Node>>,
    ) -> (Arc<Node>, Arc<Mutex<MemState>>) {
        let state = Arc::new(Mutex::new(MemState {
            data: vec![0; size as usize],
            allocated: vec![false; (size / SECTOR) as usize],
            log: Vec::new(),
        }));
        let drv = MemDriver { state: state.clone(), align, max_transfer };
        let children = backing.map(|b| vec![("backing", b)]).unwrap_or_default();
        let node =
            Node::new(name.to_string(), "mem", Box::new(drv), NodeFlags::default(), children)
                .unwrap();
        (node, state)
    }
}

impl Driver for MemDriver {
    fn pread(&self, bs: &Node, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let mut s = self.state.lock().unwrap();
        s.log.push(Op::Read(offset, buf.len() as u64));
        let backing = bs.backing().map(|c| c.node);
        let mut off = 0;
        while off < buf.len() {
            let pos = offset + off as u64;
            let n = (SECTOR - pos % SECTOR).min((buf.len() - off) as u64) as usize;
            let chunk = &mut buf[off..off + n];
            if s.allocated[(pos / SECTOR) as usize] {
                chunk.copy_from_slice(&s.data[pos as usize..pos as usize + n]);
            } else if let Some(b) = &backing {
                b.pread(pos, chunk)?;
            } else {
                chunk.fill(0);
            }
            off += n;
        }
        Ok(())
    }

    fn pwrite(&self, _: &Node, offset: u64, buf: &[u8]) -> io::Result<()> {
        let mut s = self.state.lock().unwrap();
        s.log.push(Op::Write(offset, buf.len() as u64));
        s.data[offset as usize..offset as usize + buf.len()].copy_from_slice(buf);
        let first = offset / SECTOR;
        let last = (offset + buf.len() as u64).div_ceil(SECTOR);
        for i in first..last {
            s.allocated[i as usize] = true;
        }
        Ok(())
    }

    fn getlength(&self, _: &Node) -> io::Result<u64> {
        Ok(self.state.lock().unwrap().data.len() as u64)
    }

    fn refresh_limits(&self, _: &Node, bl: &mut BlockLimits) -> Result<()> {
        bl.request_alignment = self.align;
        bl.max_transfer = self.max_transfer;
        Ok(())
    }

    fn block_status(
        &self,
        _: &Node,
        _: u32,
        offset: u64,
        bytes: u64,
    ) -> Option<io::Result<BlockStatus>> {
        let s = self.state.lock().unwrap();
        let first = (offset / SECTOR) as usize;
        let alloc = s.allocated[first];
        let mut pnum = 0;
        while pnum < bytes && s.allocated[first + (pnum / SECTOR) as usize] == alloc {
            pnum += SECTOR;
        }
        let ret = if alloc { BDRV_BLOCK_DATA } else { 0 };
        Some(Ok(BlockStatus { ret, pnum: pnum.min(bytes), map: 0, file: None }))
    }
}

fn take_log(s: &Mutex<MemState>) -> Vec<Op> {
    std::mem::take(&mut s.lock().unwrap().log)
}

/// test_threshold_not_trigger: a write below the threshold leaves it set.
#[test]
fn threshold_not_trigger() {
    let threshold = 4 * 1024 * 1024;
    let (bs, _) = MemDriver::node("wt0", 8 << 20, 512, 0);
    bs.set_write_threshold(threshold);
    bs.write_threshold_check_write(1024, 1024);
    assert_eq!(bs.write_threshold(), threshold);
}

/// test_threshold_trigger: a write past the threshold clears it.
#[test]
fn threshold_trigger() {
    let threshold = 4 * 1024 * 1024;
    let (bs, _) = MemDriver::node("wt1", 8 << 20, 512, 0);
    bs.set_write_threshold(threshold);
    bs.write_threshold_check_write(threshold - 1024, 2 * 1024);
    assert_eq!(bs.write_threshold(), 0);
}

/// The threshold is checked by real writes too, and a write that ends exactly on it does not
/// trigger it.
#[test]
fn threshold_through_writes() {
    let (bs, _) = MemDriver::node("wt2", 1 << 20, 512, 0);
    bs.set_write_threshold(65536);
    bs.pwrite(65536 - 512, &[1; 512]).unwrap();
    assert_eq!(bs.write_threshold(), 65536);
    bs.pwrite(65536, &[1; 512]).unwrap();
    assert_eq!(bs.write_threshold(), 0);
}

/// An unaligned write reads the aligned head and tail, merges, and writes whole blocks.
#[test]
fn alignment_rmw() {
    let (bs, st) = MemDriver::node("rmw", 64 << 10, 4096, 0);
    bs.pwrite(0, &[0xaa; 8192]).unwrap();
    take_log(&st);

    bs.pwrite(1000, &[0x55; 100]).unwrap();
    assert_eq!(take_log(&st), [Op::Read(0, 4096), Op::Write(0, 4096)]);
    let mut buf = vec![0; 8192];
    bs.pread(0, &mut buf).unwrap();
    assert_eq!(take_log(&st), [Op::Read(0, 8192)]);
    assert!(buf[..1000].iter().all(|&b| b == 0xaa));
    assert!(buf[1000..1100].iter().all(|&b| b == 0x55));
    assert!(buf[1100..].iter().all(|&b| b == 0xaa));

    // Crossing a block boundary: head and tail blocks are next to each other, so they are
    // read in one go (`merge_reads`), then written in one go.
    bs.pwrite(4000, &[0x11; 200]).unwrap();
    assert_eq!(take_log(&st), [Op::Read(0, 8192), Op::Write(0, 8192)]);

    // Head and tail apart: two reads, and the aligned middle is written as given.
    bs.pwrite(4000, &[0x22; 8292]).unwrap();
    assert_eq!(take_log(&st), [Op::Read(0, 4096), Op::Read(12288, 4096), Op::Write(0, 16384)]);
    bs.pwrite(4000, &[0x11; 200]).unwrap();
    take_log(&st);

    // An unaligned read reads the aligned range around it.
    let mut small = [0u8; 10];
    bs.pread(4095, &mut small).unwrap();
    assert_eq!(take_log(&st), [Op::Read(0, 8192)]);
    assert_eq!(small[0], 0x11);
    assert!(small[1..].iter().all(|&b| b == 0x11));

    // Tracked requests are gone once the requests are done.
    assert_eq!(bs.io.tracked_count(), 0);
    assert_eq!(bs.io.in_flight(), 0);
}

/// Requests longer than `max_transfer` reach the driver in pieces.
#[test]
fn max_transfer_split() {
    let (bs, st) = MemDriver::node("split", 64 << 10, 512, 8192);
    bs.pwrite(0, &[7; 20480]).unwrap();
    assert_eq!(take_log(&st), [Op::Write(0, 8192), Op::Write(8192, 8192), Op::Write(16384, 4096)]);
    let mut buf = vec![0; 20480];
    bs.pread(0, &mut buf).unwrap();
    assert_eq!(take_log(&st), [Op::Read(0, 8192), Op::Read(8192, 8192), Op::Read(16384, 4096)]);
    assert!(buf.iter().all(|&b| b == 7));
}

/// A request that overlaps a serialising one waits until that one is done.
#[test]
fn serialising_requests_wait() {
    let (bs, _) = MemDriver::node("ser", 64 << 10, 512, 0);
    let done = AtomicBool::new(false);
    let req = bs.track(0, 4096, ReqType::Write);
    bs.make_request_serialising(&req, 4096);
    std::thread::scope(|s| {
        let h = s.spawn(|| {
            bs.pwrite(1024, &[1; 512]).unwrap();
            done.store(true, Ordering::SeqCst);
        });
        // A request that does not overlap goes through.
        bs.pwrite(8192, &[2; 512]).unwrap();
        std::thread::sleep(Duration::from_millis(50));
        assert!(!done.load(Ordering::SeqCst));
        drop(req);
        h.join().unwrap();
    });
    assert!(done.load(Ordering::SeqCst));
    assert_eq!(bs.io.tracked_count(), 0);
}

/// Copy-on-read copies what a read gets from the backing file into the top image, and only
/// that.
#[test]
fn copy_on_read() {
    let (base, base_st) = MemDriver::node("cor-base", 64 << 10, 512, 0);
    base.pwrite(0, &[9; 16384]).unwrap();
    let (top, top_st) = MemDriver::node_with_backing("cor-top", 64 << 10, 512, 0, Some(base));
    top.pwrite(4096, &[3; 512]).unwrap();
    take_log(&base_st);
    take_log(&top_st);

    top.enable_copy_on_read();
    let mut buf = vec![0; 8192];
    top.pread(0, &mut buf).unwrap();
    assert!(buf[..4096].iter().all(|&b| b == 9));
    assert!(buf[4096..4608].iter().all(|&b| b == 3));
    assert!(buf[4608..].iter().all(|&b| b == 9));
    let writes: Vec<_> =
        take_log(&top_st).into_iter().filter(|o| matches!(o, Op::Write(..))).collect();
    // The allocated sector in the middle is not copied.
    assert_eq!(writes, [Op::Write(0, 4096), Op::Write(4608, 3584)]);
    top.disable_copy_on_read();

    // Everything read is now allocated in the top image.
    assert_eq!(top.is_allocated(0, 8192).unwrap(), (true, 8192));
    assert_eq!(top.is_allocated(8192, 8192).unwrap(), (false, 8192));
}

/// Block status and allocation through a backing chain.
#[test]
fn block_status_through_backing() {
    let (base, _) = MemDriver::node("bs-base", 64 << 10, 512, 0);
    base.pwrite(0, &[1; 4096]).unwrap();
    let (top, _) = MemDriver::node_with_backing("bs-top", 64 << 10, 512, 0, Some(base.clone()));
    top.pwrite(4096, &[2; 4096]).unwrap();

    // This node alone: the backing data is not allocated here.
    let st = top.block_status(0, 16384).unwrap();
    assert_eq!(st.ret & BDRV_BLOCK_ALLOCATED, 0);
    assert_eq!(st.pnum, 4096);
    let st = top.block_status(4096, 16384).unwrap();
    assert_ne!(st.ret & BDRV_BLOCK_DATA, 0);
    assert_eq!(st.pnum, 4096);

    // The whole chain.
    let st = top.block_status_above(None, 0, 16384).unwrap();
    assert_ne!(st.ret & BDRV_BLOCK_DATA, 0);
    assert_eq!(st.pnum, 4096);
    assert_eq!(top.is_allocated_above(None, false, 0, 16384).unwrap(), (2, 4096));
    assert_eq!(top.is_allocated_above(None, false, 4096, 16384).unwrap(), (1, 4096));
    assert_eq!(top.is_allocated_above(None, false, 8192, 8192).unwrap(), (0, 8192));
    // With the base as the bottom, the base does not count unless included.
    assert_eq!(top.is_allocated_above(Some(&base), false, 0, 4096).unwrap().0, 0);
    assert_eq!(top.is_allocated_above(Some(&base), true, 0, 4096).unwrap().0, 2);
}
