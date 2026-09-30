// SPDX-License-Identifier: GPL-2.0-or-later

//! The vhost-user-blk export against the vhost-user frontend of ruvm-vhost: feature and config
//! negotiation, and requests through a split ring in a file backed guest memory region, with
//! socket pairs standing in for the kick and call eventfds.

#![cfg(unix)]

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::fd::AsFd;
use std::os::unix::fs::FileExt;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use ruvm_block::{
    BLK_PERM_ALL, BLK_PERM_CONSISTENT_READ, BLK_PERM_WRITE, BlockBackend, BlockGraph,
};
use ruvm_qapi::types::{
    BlockExportOptionsVhostUserBlk, BlockdevOptions, InetSocketAddress, SocketAddress,
    SocketAddressU, UnixSocketAddress,
};
use ruvm_qapi::visit::{QObjectInputVisitor, Visit};
use ruvm_storage_daemon::export::vhost_user_blk::create;
use ruvm_storage_daemon::export::{ExportArgs, ExportDriver};
use ruvm_vhost::user::Frontend;
use ruvm_vhost::user::message::protocol;
use ruvm_vhost::{MemoryRegion, VringAddr};

const LEN: usize = 1 << 20;
const MEM: u64 = 0x10000;
/// Where the frontend pretends to have guest memory mapped.
const UADDR: u64 = 0x7f00_0000_0000;
const QSIZE: u16 = 16;
const DESC: u64 = 0;
const AVAIL: u64 = 0x100;
const USED: u64 = 0x200;

fn scratch(test: &str) -> PathBuf {
    let base = option_env!("CARGO_TARGET_TMPDIR").map_or_else(std::env::temp_dir, PathBuf::from);
    let dir = base.join("ruvm-vub").join(test);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// A graph with a `file` node `f` over a zeroed image of [`LEN`] bytes.
fn image(dir: &Path) -> (BlockGraph, PathBuf) {
    let img = dir.join("img");
    fs::write(&img, vec![0u8; LEN]).unwrap();
    let g = BlockGraph::new();
    let json =
        format!(r#"{{"driver": "file", "node-name": "f", "filename": "{}"}}"#, img.display());
    let mut v = QObjectInputVisitor::new(ruvm_qapi::json::from_str(&json).unwrap());
    let mut o = BlockdevOptions::default();
    BlockdevOptions::visit(&mut v, None, &mut o).unwrap();
    g.blockdev_add(o).unwrap();
    (g, img)
}

fn backend(g: &BlockGraph, writable: bool) -> Arc<BlockBackend> {
    let mut perm = BLK_PERM_CONSISTENT_READ;
    if writable {
        perm |= BLK_PERM_WRITE;
    }
    BlockBackend::new(g, "f", perm, BLK_PERM_ALL).unwrap()
}

fn unix(path: &Path) -> SocketAddress {
    SocketAddress { u: SocketAddressU::Unix(unix_addr(path.to_str().unwrap().to_string())) }
}

/// A `unix` address; the Linux type has more fields than the others.
fn unix_addr(path: String) -> UnixSocketAddress {
    #[cfg(target_os = "linux")]
    return UnixSocketAddress { path, abstract_: None, tight: None };
    #[cfg(not(target_os = "linux"))]
    UnixSocketAddress { path }
}

fn export(
    g: &BlockGraph,
    writable: bool,
    opts: &BlockExportOptionsVhostUserBlk,
) -> ruvm_base::Result<Arc<dyn ExportDriver>> {
    let args =
        ExportArgs { graph: g, id: "e", node_name: "f", blk: backend(g, writable), writable };
    create(&args, opts)
}

/// The driver side of one split ring in a guest memory file.
struct Driver {
    mem: File,
    kick: UnixStream,
    call: UnixStream,
    avail_idx: u16,
}

impl Driver {
    fn put(&self, gpa: u64, data: &[u8]) {
        self.mem.write_all_at(data, gpa).unwrap();
    }

    fn get(&self, gpa: u64, len: usize) -> Vec<u8> {
        let mut b = vec![0; len];
        self.mem.read_exact_at(&mut b, gpa).unwrap();
        b
    }

    /// Puts a chain of `(addr, len, device writable)` buffers on the ring, kicks the device and
    /// waits for the call.
    fn submit(&mut self, bufs: &[(u64, u32, bool)]) {
        let head = (self.avail_idx % QSIZE) * 4;
        for (i, &(addr, len, write)) in bufs.iter().enumerate() {
            let i = u16::try_from(i).unwrap() + head;
            let mut flags = if write { 2u16 } else { 0 };
            if usize::from(i - head) + 1 < bufs.len() {
                flags |= 1;
            }
            let mut d = addr.to_le_bytes().to_vec();
            d.extend_from_slice(&len.to_le_bytes());
            d.extend_from_slice(&flags.to_le_bytes());
            d.extend_from_slice(&(i + 1).to_le_bytes());
            self.put(DESC + u64::from(i) * 16, &d);
        }
        self.put(AVAIL + 4 + u64::from(self.avail_idx % QSIZE) * 2, &head.to_le_bytes());
        self.avail_idx = self.avail_idx.wrapping_add(1);
        self.put(AVAIL + 2, &self.avail_idx.to_le_bytes());
        self.kick.write_all(&1u64.to_ne_bytes()).unwrap();
        let mut b = [0u8; 8];
        self.call.read_exact(&mut b).unwrap();
        let used = u16::from_le_bytes(self.get(USED + 2, 2).try_into().unwrap());
        assert_eq!(used, self.avail_idx);
    }

    fn header(&self, ty: u32, sector: u64) {
        let mut h = ty.to_le_bytes().to_vec();
        h.extend_from_slice(&0u32.to_le_bytes());
        h.extend_from_slice(&sector.to_le_bytes());
        self.put(0x1000, &h);
    }
}

#[test]
fn bad_options() {
    let dir = scratch("bad_options");
    let (g, _) = image(&dir);
    let sock = dir.join("sock");
    let e = export(
        &g,
        true,
        &BlockExportOptionsVhostUserBlk {
            addr: unix(&sock),
            logical_block_size: Some(1000),
            num_queues: None,
        },
    )
    .err()
    .unwrap();
    assert_eq!(
        e.to_string(),
        "parameter logical-block-size must be a power of 2 between 512 and 2097152"
    );
    let e = export(
        &g,
        true,
        &BlockExportOptionsVhostUserBlk {
            addr: unix(&sock),
            logical_block_size: None,
            num_queues: Some(0),
        },
    )
    .err()
    .unwrap();
    assert_eq!(e.to_string(), "num-queues must be greater than 0");
    let inet = SocketAddress {
        u: SocketAddressU::Inet(InetSocketAddress {
            host: "127.0.0.1".into(),
            port: "0".into(),
            ..Default::default()
        }),
    };
    let e = export(
        &g,
        true,
        &BlockExportOptionsVhostUserBlk { addr: inet, logical_block_size: None, num_queues: None },
    )
    .err()
    .unwrap();
    assert_eq!(e.to_string(), "Only socket address types 'unix' and 'fd' are supported");
}

#[test]
fn negotiate_and_do_io() {
    let dir = scratch("io");
    let (g, img) = image(&dir);
    let sock = dir.join("sock");
    let exp = export(
        &g,
        true,
        &BlockExportOptionsVhostUserBlk {
            addr: unix(&sock),
            logical_block_size: Some(4096),
            num_queues: Some(2),
        },
    )
    .unwrap();
    assert!(!exp.in_use());

    let mut fe = Frontend::connect(&sock).unwrap();
    let features = fe.get_features().unwrap();
    // VIRTIO_BLK_F_RO is not offered for a writable export.
    assert_eq!(features & (1 << 5), 0);
    // Without EVENT_IDX every completion is notified.
    fe.set_features(features & !(1 << 29)).unwrap();
    let proto = fe
        .negotiate_protocol_features(protocol::CONFIG | protocol::MQ | protocol::REPLY_ACK)
        .unwrap();
    assert_eq!(proto, protocol::CONFIG | protocol::MQ | protocol::REPLY_ACK);
    fe.set_owner().unwrap();
    assert_eq!(fe.get_queue_num().unwrap(), 2);

    let cfg = fe.get_config(0, 96, 0).unwrap();
    assert_eq!(u64::from_le_bytes(cfg[0..8].try_into().unwrap()), (LEN >> 9) as u64);
    assert_eq!(u32::from_le_bytes(cfg[20..24].try_into().unwrap()), 4096);
    assert_eq!(u16::from_le_bytes(cfg[34..36].try_into().unwrap()), 2);
    fe.set_config(32, 0, &[1]).unwrap();
    assert_eq!(fe.get_config(0, 96, 0).unwrap()[32], 1);
    assert!(fe.get_config(0, 200, 0).is_err());

    let mem_path = dir.join("mem");
    fs::write(&mem_path, vec![0u8; MEM as usize]).unwrap();
    let mem = OpenOptions::new().read(true).write(true).open(&mem_path).unwrap();
    fe.set_mem_table(&[MemoryRegion {
        guest_phys_addr: 0,
        memory_size: MEM,
        userspace_addr: UADDR,
        mmap_offset: 0,
        fd: Some(mem.as_fd()),
    }])
    .unwrap();

    let (kick, kick_dev) = UnixStream::pair().unwrap();
    let (call_dev, call) = UnixStream::pair().unwrap();
    call.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    fe.set_vring_num(0, u32::from(QSIZE)).unwrap();
    fe.set_vring_addr(&VringAddr {
        index: 0,
        flags: 0,
        desc_user_addr: UADDR + DESC,
        used_user_addr: UADDR + USED,
        avail_user_addr: UADDR + AVAIL,
        log_guest_addr: 0,
    })
    .unwrap();
    fe.set_vring_base(0, 0).unwrap();
    fe.set_vring_call(0, Some(call_dev.as_fd())).unwrap();
    fe.set_vring_kick(0, Some(kick_dev.as_fd())).unwrap();
    fe.set_vring_enable(0, true).unwrap();
    drop((kick_dev, call_dev));

    let mut drv = Driver { mem, kick, call, avail_idx: 0 };
    // SET_VRING_CALL signals the new call descriptor once.
    let mut b = [0u8; 8];
    drv.call.read_exact(&mut b).unwrap();

    // VIRTIO_BLK_T_OUT of one block at sector 8.
    let data: Vec<u8> = (0..4096).map(|i| (i % 253) as u8).collect();
    drv.header(1, 8);
    drv.put(0x2000, &data);
    drv.put(0x4000, &[0xff]);
    drv.submit(&[(0x1000, 16, false), (0x2000, 4096, false), (0x4000, 1, true)]);
    assert_eq!(drv.get(0x4000, 1), [0]);
    let on_disk = fs::read(&img).unwrap();
    assert_eq!(&on_disk[4096..8192], &data[..]);

    // VIRTIO_BLK_T_IN of the same block into a zeroed buffer.
    drv.header(0, 8);
    drv.put(0x2000, &[0u8; 4096]);
    drv.put(0x4000, &[0xff]);
    drv.submit(&[(0x1000, 16, false), (0x2000, 4096, true), (0x4000, 1, true)]);
    assert_eq!(drv.get(0x4000, 1), [0]);
    assert_eq!(drv.get(0x2000, 4096), data);
    let used_len = u32::from_le_bytes(drv.get(USED + 4 + 8 + 4, 4).try_into().unwrap());
    assert_eq!(used_len, 4097);

    // An unaligned sector is an I/O error.
    drv.header(0, 1);
    drv.submit(&[(0x1000, 16, false), (0x2000, 4096, true), (0x4000, 1, true)]);
    assert_eq!(drv.get(0x4000, 1), [1]);

    // VIRTIO_BLK_T_GET_ID.
    drv.header(8, 0);
    drv.submit(&[(0x1000, 16, false), (0x2000, 20, true), (0x4000, 1, true)]);
    assert_eq!(drv.get(0x4000, 1), [0]);
    assert_eq!(&drv.get(0x2000, 14)[..], b"vhost_user_blk");

    assert_eq!(fe.get_vring_base(0).unwrap(), 4);
    drop(fe);
    exp.shutdown();
    assert!(!sock.exists());
}

#[test]
fn read_only_export() {
    let dir = scratch("ro");
    let (g, _) = image(&dir);
    let sock = dir.join("sock");
    let exp = export(
        &g,
        false,
        &BlockExportOptionsVhostUserBlk {
            addr: unix(&sock),
            logical_block_size: None,
            num_queues: None,
        },
    )
    .unwrap();
    let mut fe = Frontend::connect(&sock).unwrap();
    assert_ne!(fe.get_features().unwrap() & (1 << 5), 0);
    fe.negotiate_protocol_features(protocol::MQ).unwrap();
    assert_eq!(fe.get_queue_num().unwrap(), 1);
    drop(fe);
    // A new frontend is served once the first one has gone.
    let mut fe = Frontend::connect(&sock).unwrap();
    fe.negotiate_protocol_features(protocol::MQ).unwrap();
    assert_eq!(fe.get_queue_num().unwrap(), 1);
    drop(fe);
    exp.shutdown();
}
