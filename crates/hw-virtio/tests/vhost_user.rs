// SPDX-License-Identifier: GPL-2.0-or-later

//! vhost-user-fs and vhost-user-vsock against a fake vhost-user backend on the other end of a
//! socket pair. The fake decodes the wire format by hand, answers the requests that have
//! replies, acks `need_reply` requests and passes every message to the test, which checks the
//! sequence `hw/virtio/vhost.c` sends at init, start and stop.

#![cfg(unix)]

mod common;

use std::fs::File;
use std::io::{IoSliceMut, Read, Write};
use std::mem::MaybeUninit;
use std::os::fd::OwnedFd;
use std::os::unix::net::UnixStream;
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

use common::{Buf, Guest, RAM_SIZE};
use rustix::net::{RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags, recvmsg};
use ruvm_hw_virtio::fs::*;
use ruvm_hw_virtio::virtio::*;
use ruvm_hw_virtio::vsock::*;
use ruvm_hw_virtio::{VhostMemRegion, VhostUserChardev, VirtioDeviceClass};
use ruvm_vhost::VHOST_USER_F_PROTOCOL_FEATURES;
use ruvm_vhost::user::message::{protocol, request};

const FLAG_VERSION: u32 = 1;
const FLAG_REPLY: u32 = 1 << 2;
const FLAG_NEED_REPLY: u32 = 1 << 3;
const VRING_BASE: u32 = 42;

/// One message the fake received.
#[derive(Debug)]
struct Msg {
    request: u32,
    payload: Vec<u8>,
    fds: Vec<OwnedFd>,
}

impl Msg {
    fn u32(&self, at: usize) -> u32 {
        u32::from_ne_bytes(self.payload[at..at + 4].try_into().unwrap())
    }

    fn u64(&self, at: usize) -> u64 {
        u64::from_ne_bytes(self.payload[at..at + 8].try_into().unwrap())
    }
}

fn recv(stream: &UnixStream) -> Option<(Msg, u32)> {
    let mut header = [0u8; 12];
    let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(8))];
    let mut control = RecvAncillaryBuffer::new(&mut space);
    let mut got = 0;
    let mut fds = Vec::new();
    while got < 12 {
        let mut iov = [IoSliceMut::new(&mut header[got..])];
        let n = recvmsg(stream, &mut iov, &mut control, RecvFlags::empty()).ok()?.bytes;
        for m in control.drain() {
            if let RecvAncillaryMessage::ScmRights(r) = m {
                fds.extend(r);
            }
        }
        if n == 0 {
            return None;
        }
        got += n;
    }
    let word = |i: usize| u32::from_ne_bytes(header[i..i + 4].try_into().unwrap());
    let mut payload = vec![0u8; word(8) as usize];
    (&*stream).read_exact(&mut payload).ok()?;
    Some((Msg { request: word(0), payload, fds }, word(4)))
}

fn reply(stream: &UnixStream, request: u32, payload: &[u8]) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&request.to_ne_bytes());
    bytes.extend_from_slice(&(FLAG_VERSION | FLAG_REPLY).to_ne_bytes());
    bytes.extend_from_slice(&u32::try_from(payload.len()).unwrap().to_ne_bytes());
    bytes.extend_from_slice(payload);
    let _ = (&*stream).write_all(&bytes);
}

struct Fake {
    features: u64,
    protocol: u64,
    queues: u64,
    config: Vec<u8>,
}

impl Default for Fake {
    fn default() -> Self {
        Fake {
            features: feature(VIRTIO_F_VERSION_1) | VHOST_USER_F_PROTOCOL_FEATURES,
            protocol: protocol::MQ | protocol::REPLY_ACK | protocol::CONFIG | protocol::STATUS,
            queues: 4,
            config: 7u64.to_le_bytes().to_vec(),
        }
    }
}

impl Fake {
    /// Runs the fake on one end of a socket pair. Returns the other end and the messages as
    /// they arrive.
    fn start(self) -> (UnixStream, Receiver<Msg>) {
        let (ours, theirs) = UnixStream::pair().unwrap();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            while let Some((msg, flags)) = recv(&theirs) {
                let u64_reply = |v: u64| v.to_ne_bytes().to_vec();
                let explicit = match msg.request {
                    GET_FEATURES => Some(u64_reply(self.features)),
                    GET_PROTOCOL_FEATURES => Some(u64_reply(self.protocol)),
                    GET_QUEUE_NUM => Some(u64_reply(self.queues)),
                    GET_VRING_BASE => {
                        let mut p = msg.payload[..4].to_vec();
                        p.extend_from_slice(&VRING_BASE.to_ne_bytes());
                        Some(p)
                    }
                    request::GET_CONFIG => {
                        let offset = msg.u32(0) as usize;
                        let size = msg.u32(4) as usize;
                        let mut p = msg.payload[..12].to_vec();
                        p.extend_from_slice(&self.config[offset..offset + size]);
                        Some(p)
                    }
                    _ => None,
                };
                match explicit {
                    Some(payload) => reply(&theirs, msg.request, &payload),
                    None if flags & FLAG_NEED_REPLY != 0 => {
                        reply(&theirs, msg.request, &0u64.to_ne_bytes());
                    }
                    None => {}
                }
                if tx.send(msg).is_err() {
                    break;
                }
            }
        });
        (ours, rx)
    }
}

/// The next `n` messages.
fn take(rx: &Receiver<Msg>, n: usize) -> Vec<Msg> {
    (0..n).map(|_| rx.recv_timeout(Duration::from_secs(5)).expect("message")).collect()
}

fn requests(msgs: &[Msg]) -> Vec<u32> {
    msgs.iter().map(|m| m.request).collect()
}

fn quiet(rx: &Receiver<Msg>) {
    if let Ok(m) = rx.recv_timeout(Duration::from_millis(50)) {
        panic!("unexpected message {}", m.request);
    }
}

/// A file to stand in for guest memory in the table.
fn memory_file() -> OwnedFd {
    let path = std::env::temp_dir().join(format!(
        "ruvm-hw-virtio-vhost-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let file = File::options().read(true).write(true).create(true).truncate(true).open(&path);
    let file = file.unwrap();
    std::fs::remove_file(&path).unwrap();
    file.set_len(RAM_SIZE).unwrap();
    OwnedFd::from(file)
}

const RAM_UVA: u64 = 0x7f00_0000_0000;

fn mem_table() -> Vec<VhostMemRegion> {
    vec![VhostMemRegion {
        guest_phys_addr: 0,
        memory_size: RAM_SIZE,
        userspace_addr: RAM_UVA,
        mmap_offset: 0,
        fd: Some(Arc::new(memory_file())),
    }]
}

fn fs_conf(stream: UnixStream) -> VhostUserFsConf {
    VhostUserFsConf {
        chardev: Some(VhostUserChardev::Stream(stream)),
        tag: Some("myfs".into()),
        num_request_queues: 1,
        queue_size: 128,
    }
}

fn realize_err(dev: Box<dyn VirtioDeviceClass>) -> String {
    match Guest::try_new(false, Some(dev)) {
        Ok(_) => panic!("realize should fail"),
        Err(e) => e.to_string(),
    }
}

use request::{
    GET_FEATURES, GET_PROTOCOL_FEATURES, GET_QUEUE_NUM, GET_VRING_BASE, SET_FEATURES,
    SET_MEM_TABLE, SET_OWNER, SET_PROTOCOL_FEATURES, SET_VRING_ADDR, SET_VRING_BASE,
    SET_VRING_CALL, SET_VRING_ENABLE, SET_VRING_ERR, SET_VRING_KICK, SET_VRING_NUM,
};

#[test]
fn fs_message_sequence() {
    let (stream, rx) = Fake::default().start();
    let mut g = Guest::new(false, Some(Box::new(VhostUserFs::new(fs_conf(stream)))));

    // vhost_user_backend_init(), then vhost_dev_init().
    let init = take(&rx, 10);
    assert_eq!(
        requests(&init),
        [
            GET_FEATURES,
            GET_PROTOCOL_FEATURES,
            SET_PROTOCOL_FEATURES,
            GET_QUEUE_NUM,
            SET_OWNER,
            GET_FEATURES,
            SET_VRING_CALL,
            SET_VRING_ERR,
            SET_VRING_CALL,
            SET_VRING_ERR,
        ]
    );
    // Only MQ and REPLY_ACK of what the fake offers: no CONFIG for virtio-fs, no STATUS.
    assert_eq!(init[2].u64(0), protocol::MQ | protocol::REPLY_ACK);
    assert_eq!(init[6].u64(0), 0);
    assert_eq!(init[7].u64(0), 0);
    assert_eq!(init[8].u64(0), 1);
    assert_eq!(init[6].fds.len(), 1);
    quiet(&rx);

    // Config space: the tag with its terminator, then num_request_queues.
    assert_eq!(g.readl(ruvm_hw_virtio::mmio::VIRTIO_MMIO_DEVICE_ID), 26);
    let tag: Vec<u8> = (0..8).map(|i| g.config_readb(i)).collect();
    assert_eq!(&tag, b"myfs\0\0\0\0");
    assert_eq!(g.config_readl(36), 1);

    let accepted = g.negotiate(!0);
    assert_ne!(accepted & feature(VIRTIO_F_VERSION_1), 0);
    let hiprio = g.setup_queue(0, 0);
    let mut req = g.setup_queue(1, 64);
    g.mmio.with_device(|_, d: &mut VhostUserFs| d.set_mem_table(mem_table())).unwrap().unwrap();
    g.driver_ok();

    // vhost_dev_start().
    let start = take(&rx, 14);
    assert_eq!(
        requests(&start),
        [
            SET_FEATURES,
            SET_MEM_TABLE,
            SET_VRING_NUM,
            SET_VRING_BASE,
            SET_VRING_ADDR,
            SET_VRING_KICK,
            SET_VRING_CALL,
            SET_VRING_NUM,
            SET_VRING_BASE,
            SET_VRING_ADDR,
            SET_VRING_KICK,
            SET_VRING_CALL,
            SET_VRING_ENABLE,
            SET_VRING_ENABLE,
        ]
    );
    // The driver's features, the protocol bit kept, nothing the backend did not offer.
    assert_eq!(start[0].u64(0), feature(VIRTIO_F_VERSION_1) | VHOST_USER_F_PROTOCOL_FEATURES);
    assert_eq!(start[1].fds.len(), 1);
    assert_eq!((start[2].u32(0), start[2].u32(4)), (0, 128));
    assert_eq!((start[7].u32(0), start[7].u32(4)), (1, 64));
    assert_eq!(start[3].u32(4), 0);
    // SET_VRING_ADDR: index, flags, then desc, used, avail and log addresses.
    let addr = &start[9];
    assert_eq!(addr.u32(0), 1);
    assert_eq!(addr.u64(8), RAM_UVA + req.desc);
    assert_eq!(addr.u64(16), RAM_UVA + req.used);
    assert_eq!(addr.u64(24), RAM_UVA + req.avail);
    assert_eq!(addr.u64(32), req.used);
    assert_eq!(start[4].u64(8), RAM_UVA + hiprio.desc);
    assert_eq!((start[12].u32(0), start[12].u32(4)), (0, 1));
    assert_eq!((start[13].u32(0), start[13].u32(4)), (1, 1));
    quiet(&rx);

    // A guest kick goes to the backend through the kick file.
    let mut start = start;
    let kick = File::from(start[10].fds.pop().expect("SET_VRING_KICK carries a file"));
    let a = g.alloc(64, 16);
    req.submit(&g, &[Buf::out(a, 64)]);
    g.kick(&req);
    let mut buf = [0u8; 8];
    (&kick).read_exact(&mut buf).unwrap();
    assert_eq!(u64::from_ne_bytes(buf), 1);

    // The backend signals the call file: polling turns that into an interrupt.
    let call = File::from(start[11].fds.pop().expect("SET_VRING_CALL carries a file"));
    assert!(!g.irq());
    (&call).write_all(&1u64.to_ne_bytes()).unwrap();
    let polled = g.mmio.with_device(|vdev, d: &mut VhostUserFs| d.poll_calls(vdev)).unwrap();
    assert!(polled);
    assert!(g.irq());
    assert_eq!(g.isr() & 1, 1);

    // vhost_dev_stop(): rings disabled, then their state fetched back.
    g.set_status(0);
    let stop = take(&rx, 4);
    assert_eq!(
        requests(&stop),
        [SET_VRING_ENABLE, SET_VRING_ENABLE, GET_VRING_BASE, GET_VRING_BASE]
    );
    assert_eq!((stop[0].u32(0), stop[0].u32(4)), (0, 0));
    assert_eq!(stop[3].u32(0), 1);
    quiet(&rx);
    g.mmio.with_device(|_, d: &mut VhostUserFs| assert!(!d.vhost().unwrap().is_started())).unwrap();
}

#[test]
fn fs_resumes_from_the_backend_ring_state() {
    let (stream, rx) = Fake::default().start();
    let g = Guest::new(false, Some(Box::new(VhostUserFs::new(fs_conf(stream)))));
    let _ = take(&rx, 10);
    g.negotiate(!0);
    let mut g = g;
    let _ = g.setup_queue(0, 0);
    let _ = g.setup_queue(1, 0);
    g.mmio.with_device(|_, d: &mut VhostUserFs| d.set_mem_table(mem_table())).unwrap().unwrap();
    g.driver_ok();
    let _ = take(&rx, 14);
    // Stop through the device, as a reset would, and look at the queue state before the core
    // resets the queues.
    g.mmio
        .with_device(|vdev, d: &mut VhostUserFs| {
            d.set_status(vdev, 0).unwrap();
            assert_eq!(vdev.last_avail_idx(0), VRING_BASE);
            assert_eq!(vdev.last_avail_idx(1), VRING_BASE);
        })
        .unwrap();
    assert_eq!(requests(&take(&rx, 4))[2..], [GET_VRING_BASE, GET_VRING_BASE]);
}

#[test]
fn fs_without_mem_table_fails_to_start() {
    let (stream, rx) = Fake::default().start();
    let mut g = Guest::new(false, Some(Box::new(VhostUserFs::new(fs_conf(stream)))));
    let _ = take(&rx, 10);
    g.negotiate(!0);
    let _ = g.setup_queue(0, 0);
    g.driver_ok();
    // SET_FEATURES and an empty SET_MEM_TABLE go out, then the ring cannot be mapped.
    assert_eq!(requests(&take(&rx, 2)), [SET_FEATURES, SET_MEM_TABLE]);
    quiet(&rx);
    g.mmio.with_device(|_, d: &mut VhostUserFs| assert!(!d.vhost().unwrap().is_started())).unwrap();
}

#[test]
fn fs_feature_masking() {
    let fake = Fake { features: VHOST_USER_F_PROTOCOL_FEATURES, protocol: 0, ..Fake::default() };
    let (stream, rx) = fake.start();
    let g = Guest::new(false, Some(Box::new(VhostUserFs::new(fs_conf(stream)))));
    // Without protocol features there is no GET_QUEUE_NUM and no SET_VRING_ENABLE.
    let init = take(&rx, 9);
    assert_eq!(
        requests(&init)[..5],
        [GET_FEATURES, GET_PROTOCOL_FEATURES, SET_PROTOCOL_FEATURES, SET_OWNER, GET_FEATURES]
    );
    // VERSION_1 is in user_feature_bits and the backend lacks it, so it is gone, as are the
    // other masked ring features.
    let f = g.device_features();
    assert_eq!(f & feature(VIRTIO_F_VERSION_1), 0);
    assert_eq!(f & feature(29), 0, "EVENT_IDX");
    assert_eq!(f & feature(28), 0, "INDIRECT_DESC");
}

#[test]
fn fs_realize_errors() {
    let fs = |f: &dyn Fn(&mut VhostUserFsConf)| {
        let (stream, _rx) = Fake::default().start();
        let mut conf = fs_conf(stream);
        f(&mut conf);
        realize_err(Box::new(VhostUserFs::new(conf)))
    };
    assert_eq!(fs(&|c| c.chardev = None), "missing chardev");
    assert_eq!(fs(&|c| c.tag = None), "missing tag property");
    assert_eq!(fs(&|c| c.tag = Some(String::new())), "tag property cannot be empty");
    assert_eq!(fs(&|c| c.tag = Some("x".repeat(37))), "tag property must be 36 bytes or less");
    assert_eq!(
        fs(&|c| c.num_request_queues = 0),
        "num-request-queues property must be larger than 0"
    );
    assert_eq!(fs(&|c| c.queue_size = 100), "queue-size property must be a power of 2");
    assert_eq!(fs(&|c| c.queue_size = 2048), "queue-size property must be 1024 or smaller");

    let path = std::env::temp_dir().join("ruvm-no-such-vhost-user-socket");
    let conf = VhostUserFsConf {
        chardev: Some(VhostUserChardev::Path(path.clone())),
        ..fs_conf(UnixStream::pair().unwrap().0)
    };
    assert_eq!(
        realize_err(Box::new(VhostUserFs::new(conf))),
        format!("Failed to connect to '{}': No such file or directory", path.display())
    );
}

#[test]
fn fs_tag_of_36_bytes_has_no_terminator() {
    let (stream, _rx) = Fake::default().start();
    let tag = "t".repeat(36);
    let conf = VhostUserFsConf { tag: Some(tag.clone()), num_request_queues: 3, ..fs_conf(stream) };
    let g = Guest::new(false, Some(Box::new(VhostUserFs::new(conf))));
    let got: Vec<u8> = (0..36).map(|i| g.config_readb(i)).collect();
    assert_eq!(got, tag.as_bytes());
    assert_eq!(g.config_readl(36), 3);
    g.mmio
        .with_device(|vdev, d: &mut VhostUserFs| {
            assert_eq!(vdev.num_queues(), 4);
            assert_eq!(d.vhost().unwrap().nvqs(), 4);
            assert_eq!(d.vhost().unwrap().max_queues(), 4);
        })
        .unwrap();
}

#[test]
fn fs_hang_up_during_init() {
    let (ours, theirs) = UnixStream::pair().unwrap();
    drop(theirs);
    let err = realize_err(Box::new(VhostUserFs::new(fs_conf(ours))));
    assert!(err.starts_with("vhost_backend_init failed: "), "{err}");
}

fn user_vsock(stream: UnixStream, seqpacket: OnOffAuto) -> Box<dyn VirtioDeviceClass> {
    Box::new(VhostUserVsock::new(VhostUserVsockConf {
        chardev: Some(VhostUserChardev::Stream(stream)),
        seqpacket,
    }))
}

#[test]
fn user_vsock_reads_config_from_the_backend() {
    let (stream, rx) = Fake::default().start();
    let mut g = Guest::new(false, Some(user_vsock(stream, OnOffAuto::Auto)));
    let init = take(&rx, 11);
    assert_eq!(
        requests(&init),
        [
            GET_FEATURES,
            GET_PROTOCOL_FEATURES,
            SET_PROTOCOL_FEATURES,
            GET_QUEUE_NUM,
            SET_OWNER,
            GET_FEATURES,
            SET_VRING_CALL,
            SET_VRING_ERR,
            SET_VRING_CALL,
            SET_VRING_ERR,
            request::GET_CONFIG,
        ]
    );
    assert_eq!(init[2].u64(0), protocol::MQ | protocol::REPLY_ACK | protocol::CONFIG);
    assert_eq!(g.readl(ruvm_hw_virtio::mmio::VIRTIO_MMIO_DEVICE_ID), 19);
    assert_eq!(g.config_readq(0), 7);
    // The backend offers no seqpacket, so auto leaves it off.
    assert_eq!(g.device_features() & feature(VIRTIO_VSOCK_F_SEQPACKET), 0);

    g.negotiate(!0);
    let _ = g.setup_queue(0, 0);
    let _ = g.setup_queue(1, 0);
    let mut event = g.setup_queue(2, 0);
    g.mmio.with_device(|_, d: &mut VhostUserVsock| d.set_mem_table(mem_table())).unwrap().unwrap();
    g.driver_ok();
    // Two rings: the event queue stays with the device.
    let start = take(&rx, 14);
    assert_eq!(start[0].request, SET_FEATURES);
    assert_eq!(requests(&start)[12..], [SET_VRING_ENABLE, SET_VRING_ENABLE]);
    quiet(&rx);

    // The transport reset event.
    let a = g.alloc(4, 4);
    g.write_mem(a, &[0xee; 4]);
    event.submit(&g, &[Buf::inp(a, 4)]);
    g.mmio.with_device(|vdev, d: &mut VhostUserVsock| d.send_transport_reset(vdev)).unwrap();
    assert_eq!(event.get_used(&g).map(|(_, len)| len), Some(4));
    assert_eq!(g.read_u32(a), 0);
    // A buffer with a device readable part is dropped.
    event.submit(&g, &[Buf::out(a, 4)]);
    g.mmio.with_device(|vdev, d: &mut VhostUserVsock| d.send_transport_reset(vdev)).unwrap();
    assert!(event.get_used(&g).is_none());

    g.set_status(0);
    assert_eq!(requests(&take(&rx, 4))[2..], [GET_VRING_BASE, GET_VRING_BASE]);
}

#[test]
fn user_vsock_needs_config_protocol_feature() {
    let fake = Fake { protocol: protocol::MQ, ..Fake::default() };
    let (stream, _rx) = fake.start();
    assert_eq!(
        realize_err(user_vsock(stream, OnOffAuto::Auto)),
        "vhost-user device expecting VHOST_USER_PROTOCOL_F_CONFIG but the vhost-user backend \
         does not support it."
    );
    assert_eq!(
        realize_err(Box::new(VhostUserVsock::new(VhostUserVsockConf::default()))),
        "missing chardev"
    );
}

#[test]
fn user_vsock_seqpacket() {
    let with_seqpacket = Fake {
        features: feature(VIRTIO_F_VERSION_1)
            | VHOST_USER_F_PROTOCOL_FEATURES
            | feature(VIRTIO_VSOCK_F_SEQPACKET),
        ..Fake::default()
    };
    let (stream, _rx) = with_seqpacket.start();
    let g = Guest::new(false, Some(user_vsock(stream, OnOffAuto::On)));
    assert_ne!(g.device_features() & feature(VIRTIO_VSOCK_F_SEQPACKET), 0);

    let (stream, _rx) = Fake::default().start();
    let err = realize_err(user_vsock(stream, OnOffAuto::On));
    assert!(err.ends_with("vhost-vsock backend doesn't support seqpacket"), "{err}");
}

#[cfg(not(target_os = "linux"))]
#[test]
fn kernel_vsock_is_missing() {
    let dev = |cid| {
        let conf = VhostVsockConf { guest_cid: cid, ..VhostVsockConf::default() };
        realize_err(Box::new(VhostVsock::new(conf)))
    };
    assert_eq!(dev(2), "guest-cid property must be greater than 2");
    assert_eq!(dev(1 << 32), "guest-cid property must be a 32-bit number");
    assert_eq!(dev(3), "Could not open '/dev/vhost-vsock': No such file or directory");
}
