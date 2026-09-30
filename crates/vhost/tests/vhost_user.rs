// SPDX-License-Identifier: MIT OR Apache-2.0

//! The vhost-user frontend against an in-process fake backend on the other end of a socket pair.
//!
//! The fake decodes the wire format by hand rather than through the crate, so a mistake in the
//! crate's encoding shows up as a mismatch here instead of cancelling itself out. Its default
//! behaviour follows the test server in QEMU's tests/qtest/vhost-user-test.c: it offers
//! `VIRTIO_F_VERSION_1`, `VHOST_F_LOG_ALL` and protocol features, answers the requests that have
//! replies, acks `need_reply` requests, and records everything it was sent.

#![cfg(unix)]

use std::fs::File;
use std::io::{IoSliceMut, Read, Write};
use std::mem::MaybeUninit;
use std::os::fd::{AsFd, OwnedFd};
use std::os::unix::fs::FileExt;
use std::os::unix::net::UnixStream;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use rustix::net::{RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags, recvmsg};
use ruvm_vhost::user::message::{protocol, request};
use ruvm_vhost::user::{BackendRequest, Frontend};
use ruvm_vhost::{
    Error, LogRegion, MemoryRegion, VHOST_F_LOG_ALL, VHOST_USER_F_PROTOCOL_FEATURES, VhostBackend,
    VringAddr,
};

const VIRTIO_F_VERSION_1: u64 = 1 << 32;
const FLAG_VERSION: u32 = 1;
const FLAG_REPLY: u32 = 1 << 2;
const FLAG_NEED_REPLY: u32 = 1 << 3;

/// One message the fake received.
#[derive(Debug)]
struct Msg {
    request: u32,
    flags: u32,
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

/// Read one message, or `None` at end of stream.
fn recv(stream: &UnixStream) -> Option<Msg> {
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
    Some(Msg { request: word(0), flags: word(4), payload, fds })
}

fn send_raw(stream: &UnixStream, request: u32, flags: u32, payload: &[u8]) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&request.to_ne_bytes());
    bytes.extend_from_slice(&flags.to_ne_bytes());
    bytes.extend_from_slice(&u32::try_from(payload.len()).unwrap().to_ne_bytes());
    bytes.extend_from_slice(payload);
    (&*stream).write_all(&bytes).unwrap();
}

fn reply(stream: &UnixStream, request: u32, payload: &[u8]) {
    send_raw(stream, request, FLAG_VERSION | FLAG_REPLY, payload);
}

type Hook = Box<dyn FnMut(&UnixStream, &Msg) -> Hooked + Send>;

/// What a test hook did with a message.
enum Hooked {
    /// Nothing; let the fake handle it.
    Pass,
    /// Replied already; just record it.
    Handled,
    /// Close the connection.
    HangUp,
}

struct Fake {
    features: u64,
    protocol: u64,
    queues: u64,
    ack_status: u64,
    config: Vec<u8>,
    hook: Option<Hook>,
    backend_fd: Option<mpsc::Sender<OwnedFd>>,
}

impl Default for Fake {
    fn default() -> Self {
        Fake {
            features: VIRTIO_F_VERSION_1 | VHOST_F_LOG_ALL | VHOST_USER_F_PROTOCOL_FEATURES,
            protocol: protocol::LOG_SHMFD | protocol::CROSS_ENDIAN,
            queues: 1,
            ack_status: 0,
            config: (0u8..64).collect(),
            hook: None,
            backend_fd: None,
        }
    }
}

impl Fake {
    fn with_protocol(protocol: u64) -> Self {
        Fake { protocol, ..Fake::default() }
    }

    /// Start the fake on one end of a socket pair and return a frontend on the other, plus a
    /// handle that yields everything the fake received once the frontend is dropped.
    fn start(mut self) -> (Frontend, JoinHandle<Vec<Msg>>) {
        let (ours, theirs) = UnixStream::pair().unwrap();
        let handle = std::thread::spawn(move || {
            let mut log = Vec::new();
            let mut status = 0u64;
            while let Some(mut msg) = recv(&theirs) {
                if let Some(hook) = self.hook.as_mut() {
                    match hook(&theirs, &msg) {
                        Hooked::Pass => {}
                        Hooked::Handled => {
                            log.push(msg);
                            continue;
                        }
                        Hooked::HangUp => {
                            log.push(msg);
                            break;
                        }
                    }
                }
                let u64_reply = |v: u64| v.to_ne_bytes().to_vec();
                let explicit = match msg.request {
                    request::GET_FEATURES => Some(u64_reply(self.features)),
                    request::GET_PROTOCOL_FEATURES => Some(u64_reply(self.protocol)),
                    request::GET_QUEUE_NUM => Some(u64_reply(self.queues)),
                    request::GET_MAX_MEM_SLOTS => Some(u64_reply(32)),
                    request::GET_STATUS => Some(u64_reply(status)),
                    request::GET_VRING_BASE => {
                        let mut p = msg.payload[..4].to_vec();
                        p.extend_from_slice(&42u32.to_ne_bytes());
                        Some(p)
                    }
                    request::GET_CONFIG => {
                        let offset = msg.u32(0) as usize;
                        let size = msg.u32(4) as usize;
                        let mut p = msg.payload[..12].to_vec();
                        p.extend_from_slice(&self.config[offset..offset + size]);
                        Some(p)
                    }
                    request::SET_LOG_BASE if !msg.fds.is_empty() => Some(Vec::new()),
                    _ => None,
                };
                if msg.request == request::SET_STATUS {
                    status = msg.u64(0);
                }
                if msg.request == request::SET_BACKEND_REQ_FD {
                    if let Some(tx) = &self.backend_fd {
                        tx.send(msg.fds.remove(0)).unwrap();
                    }
                }
                match explicit {
                    Some(payload) => reply(&theirs, msg.request, &payload),
                    None if msg.flags & FLAG_NEED_REPLY != 0 => {
                        reply(&theirs, msg.request, &self.ack_status.to_ne_bytes());
                    }
                    None => {}
                }
                log.push(msg);
            }
            log
        });
        (Frontend::new(ours), handle)
    }
}

fn finish(frontend: Frontend, handle: JoinHandle<Vec<Msg>>) -> Vec<Msg> {
    drop(frontend);
    handle.join().unwrap()
}

fn requests(log: &[Msg]) -> Vec<u32> {
    log.iter().map(|m| m.request).collect()
}

/// A file to back guest memory: a memfd on Linux, an unlinked temporary file elsewhere.
fn memory_file(size: u64) -> File {
    #[cfg(target_os = "linux")]
    let file = File::from(
        rustix::fs::memfd_create("vhost-user-test", rustix::fs::MemfdFlags::CLOEXEC).unwrap(),
    );
    #[cfg(not(target_os = "linux"))]
    let file = {
        use std::sync::atomic::{AtomicU32, Ordering};
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let path = std::env::temp_dir().join(format!(
            "ruvm-vhost-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let file = File::options().read(true).write(true).create_new(true).open(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        file
    };
    file.set_len(size).unwrap();
    file
}

fn negotiated(fake: Fake, supported: u64) -> (Frontend, JoinHandle<Vec<Msg>>) {
    let (mut frontend, handle) = fake.start();
    frontend.get_features().unwrap();
    frontend.negotiate_protocol_features(supported).unwrap();
    (frontend, handle)
}

#[test]
fn feature_negotiation_follows_the_qtest_server() {
    let (mut frontend, handle) = Fake::default().start();
    let features = frontend.get_features().unwrap();
    assert_eq!(features, VIRTIO_F_VERSION_1 | VHOST_F_LOG_ALL | VHOST_USER_F_PROTOCOL_FEATURES);
    let agreed = frontend
        .negotiate_protocol_features(protocol::LOG_SHMFD | protocol::MQ | protocol::REPLY_ACK)
        .unwrap();
    assert_eq!(agreed, protocol::LOG_SHMFD);
    assert_eq!(frontend.acked_protocol_features(), protocol::LOG_SHMFD);
    frontend.set_owner().unwrap();
    frontend.set_features(VIRTIO_F_VERSION_1).unwrap();
    assert!(matches!(frontend.set_features(1 << 5), Err(Error::InvalidArgument(_))));

    let log = finish(frontend, handle);
    assert_eq!(
        requests(&log),
        [
            request::GET_FEATURES,
            request::GET_PROTOCOL_FEATURES,
            request::SET_PROTOCOL_FEATURES,
            request::SET_OWNER,
            request::SET_FEATURES,
        ]
    );
    for msg in &log {
        assert_eq!(msg.flags, FLAG_VERSION, "request {}", msg.request);
    }
    assert!(log[0].payload.is_empty());
    assert_eq!(log[2].u64(0), protocol::LOG_SHMFD);
    // vhost-user-test.c asserts that SET_FEATURES keeps VHOST_USER_F_PROTOCOL_FEATURES.
    assert_eq!(log[4].u64(0), VIRTIO_F_VERSION_1 | VHOST_USER_F_PROTOCOL_FEATURES);
}

#[test]
fn backend_without_protocol_features() {
    // The qtest server's "bad" mode answers GET_FEATURES with zero.
    let (mut frontend, handle) = Fake { features: 0, ..Fake::default() }.start();
    assert_eq!(frontend.negotiate_protocol_features(u64::MAX).unwrap(), 0);
    assert!(matches!(frontend.get_protocol_features(), Err(Error::NotNegotiated(_))));
    assert!(matches!(frontend.set_vring_enable(0, true), Err(Error::NotNegotiated(_))));
    frontend.set_features(0).unwrap();
    let log = finish(frontend, handle);
    assert_eq!(requests(&log), [request::GET_FEATURES, request::SET_FEATURES]);
    assert_eq!(log[1].u64(0), 0);
}

#[test]
fn protocol_features_gate_requests() {
    let (mut frontend, handle) = negotiated(Fake::default(), u64::MAX);
    assert!(matches!(frontend.get_queue_num(), Err(Error::NotNegotiated(_))));
    assert!(matches!(frontend.get_config(0, 4, 0), Err(Error::NotNegotiated(_))));
    assert!(matches!(frontend.get_max_mem_slots(), Err(Error::NotNegotiated(_))));
    assert!(matches!(frontend.get_status(), Err(Error::NotNegotiated(_))));
    assert!(matches!(frontend.reset_device(), Err(Error::NotNegotiated(_))));
    assert!(matches!(frontend.set_backend_req_fd(Box::new(|_| 0)), Err(Error::NotNegotiated(_))));
    assert!(matches!(
        frontend.set_protocol_features(protocol::STATUS),
        Err(Error::InvalidArgument(_))
    ));
    let log = finish(frontend, handle);
    assert_eq!(log.len(), 3, "nothing gated reached the backend");

    // The qtest multiqueue server offers MQ and reports its queue count.
    let fake = Fake { queues: 2, ..Fake::with_protocol(protocol::MQ) };
    let (mut frontend, handle) = negotiated(fake, protocol::MQ);
    assert_eq!(frontend.get_queue_num().unwrap(), 2);
    finish(frontend, handle);
}

#[test]
fn inband_notifications_need_backend_req_and_reply_ack() {
    let offered = protocol::INBAND_NOTIFICATIONS | protocol::REPLY_ACK;
    let (frontend, handle) = negotiated(Fake::with_protocol(offered), u64::MAX);
    assert_eq!(frontend.acked_protocol_features(), protocol::REPLY_ACK);
    finish(frontend, handle);

    let offered = protocol::INBAND_NOTIFICATIONS | protocol::REPLY_ACK | protocol::BACKEND_REQ;
    let (frontend, handle) = negotiated(Fake::with_protocol(offered), u64::MAX);
    assert_eq!(frontend.acked_protocol_features(), offered);
    finish(frontend, handle);
}

#[test]
fn mem_table_passes_one_fd_per_region() {
    let (mut frontend, handle) = negotiated(Fake::default(), 0);
    let low = memory_file(0x10000);
    let high = memory_file(0x20000);
    low.write_all_at(b"low memory", 0).unwrap();
    high.write_all_at(b"high memory", 0x1000).unwrap();
    let regions = [
        MemoryRegion {
            guest_phys_addr: 0,
            memory_size: 0x10000,
            userspace_addr: 0x7f00_0000_0000,
            mmap_offset: 0,
            fd: Some(low.as_fd()),
        },
        MemoryRegion {
            guest_phys_addr: 0x10_0000,
            memory_size: 0x1f000,
            userspace_addr: 0x7f00_1000_0000,
            mmap_offset: 0x1000,
            fd: Some(high.as_fd()),
        },
    ];
    frontend.set_mem_table(&regions).unwrap();
    let log = finish(frontend, handle);
    let msg = log.iter().find(|m| m.request == request::SET_MEM_TABLE).unwrap();
    assert_eq!(msg.payload.len(), 8 + 2 * 32);
    assert_eq!(msg.u32(0), 2);
    assert_eq!(msg.u32(4), 0);
    // vhost-user-test.c checks the fd count against nregions and that each region is big.
    assert_eq!(msg.fds.len(), 2);
    for (i, region) in regions.iter().enumerate() {
        let at = 8 + i * 32;
        assert_eq!(msg.u64(at), region.guest_phys_addr);
        assert_eq!(msg.u64(at + 8), region.memory_size);
        assert!(msg.u64(at + 8) > 1024);
        assert_eq!(msg.u64(at + 16), region.userspace_addr);
        assert_eq!(msg.u64(at + 24), region.mmap_offset);
    }
    // The descriptors are the same files: read the markers back through them.
    let mut buf = [0u8; 11];
    let file = File::from(msg.fds[0].try_clone().unwrap());
    file.read_exact_at(&mut buf[..10], msg.u64(8 + 24)).unwrap();
    assert_eq!(&buf[..10], b"low memory");
    let file = File::from(msg.fds[1].try_clone().unwrap());
    file.read_exact_at(&mut buf, msg.u64(8 + 32 + 24)).unwrap();
    assert_eq!(&buf, b"high memory");
}

#[test]
fn mem_table_limits() {
    let (mut frontend, handle) = negotiated(Fake::default(), 0);
    let file = memory_file(0x1000);
    let region = |i: u64| MemoryRegion {
        guest_phys_addr: i * 0x1000,
        memory_size: 0x1000,
        userspace_addr: 0x1000_0000 + i * 0x1000,
        mmap_offset: 0,
        fd: Some(file.as_fd()),
    };
    let nine: Vec<_> = (0..9).map(region).collect();
    assert!(matches!(
        frontend.set_mem_table(&nine),
        Err(Error::TooManyRegions { count: 9, max: 8 })
    ));
    frontend.set_mem_table(&nine[..8]).unwrap();
    let no_fd = [MemoryRegion { fd: None, ..region(0) }];
    assert!(matches!(frontend.set_mem_table(&no_fd), Err(Error::InvalidArgument(_))));
    let overlap = [region(0), MemoryRegion { guest_phys_addr: 0x800, ..region(5) }];
    assert!(matches!(frontend.set_mem_table(&overlap), Err(Error::InvalidArgument(_))));
    let log = finish(frontend, handle);
    let tables: Vec<_> = log.iter().filter(|m| m.request == request::SET_MEM_TABLE).collect();
    assert_eq!(tables.len(), 1);
    assert_eq!(tables[0].fds.len(), 8);
    assert_eq!(tables[0].u32(0), 8);
}

#[test]
fn add_and_remove_memory_regions() {
    let fake = Fake::with_protocol(protocol::CONFIGURE_MEM_SLOTS);
    let (mut frontend, handle) = negotiated(fake, protocol::CONFIGURE_MEM_SLOTS);
    assert_eq!(frontend.get_max_mem_slots().unwrap(), 32);
    let file = memory_file(0x2000);
    let region = MemoryRegion {
        guest_phys_addr: 0x4000_0000,
        memory_size: 0x2000,
        userspace_addr: 0x7000_0000,
        mmap_offset: 0x1000,
        fd: Some(file.as_fd()),
    };
    frontend.add_mem_reg(&region).unwrap();
    frontend.rem_mem_reg(&region).unwrap();
    let log = finish(frontend, handle);
    let add = log.iter().find(|m| m.request == request::ADD_MEM_REG).unwrap();
    let rem = log.iter().find(|m| m.request == request::REM_MEM_REG).unwrap();
    for msg in [add, rem] {
        assert_eq!(msg.payload.len(), 40);
        assert_eq!(msg.u64(0), 0, "padding");
        assert_eq!(msg.u64(8), 0x4000_0000);
        assert_eq!(msg.u64(16), 0x2000);
        assert_eq!(msg.u64(24), 0x7000_0000);
        assert_eq!(msg.u64(32), 0x1000);
    }
    assert_eq!(add.fds.len(), 1);
    assert!(rem.fds.is_empty());
}

#[test]
fn vring_setup() {
    let (mut frontend, handle) = negotiated(Fake::default(), 0);
    let (kick, _kick_peer) = UnixStream::pair().unwrap();
    let (call, _call_peer) = UnixStream::pair().unwrap();
    let addr = VringAddr {
        index: 1,
        flags: 0,
        desc_user_addr: 0x1000,
        used_user_addr: 0x3000,
        avail_user_addr: 0x2000,
        log_guest_addr: 0x9000,
    };
    frontend.set_vring_num(1, 256).unwrap();
    frontend.set_vring_addr(&addr).unwrap();
    frontend.set_vring_base(1, 7).unwrap();
    frontend.set_vring_call(1, Some(call.as_fd())).unwrap();
    frontend.set_vring_kick(1, Some(kick.as_fd())).unwrap();
    frontend.set_vring_err(1, None).unwrap();
    frontend.set_vring_enable(1, true).unwrap();
    assert_eq!(frontend.get_vring_base(1).unwrap(), 42);
    assert!(matches!(frontend.set_vring_kick(256, None), Err(Error::InvalidArgument(_))));

    let log = finish(frontend, handle);
    let log = &log[3..];
    assert_eq!(
        requests(log),
        [
            request::SET_VRING_NUM,
            request::SET_VRING_ADDR,
            request::SET_VRING_BASE,
            request::SET_VRING_CALL,
            request::SET_VRING_KICK,
            request::SET_VRING_ERR,
            request::SET_VRING_ENABLE,
            request::GET_VRING_BASE,
        ]
    );
    assert_eq!((log[0].u32(0), log[0].u32(4)), (1, 256));
    assert_eq!(log[1].payload.len(), 40);
    assert_eq!((log[1].u32(0), log[1].u32(4)), (1, 0));
    assert_eq!((log[1].u64(8), log[1].u64(16), log[1].u64(24)), (0x1000, 0x3000, 0x2000));
    assert_eq!(log[1].u64(32), 0x9000);
    assert_eq!((log[2].u32(0), log[2].u32(4)), (1, 7));
    assert_eq!((log[3].u64(0), log[3].fds.len()), (1, 1));
    assert_eq!((log[4].u64(0), log[4].fds.len()), (1, 1));
    // No descriptor: bit 8 says so.
    assert_eq!((log[5].u64(0), log[5].fds.len()), (1 | 0x100, 0));
    assert_eq!((log[6].u32(0), log[6].u32(4)), (1, 1));
    // GET_VRING_BASE's num field is reserved and must be zero.
    assert_eq!((log[7].u32(0), log[7].u32(4)), (1, 0));
}

#[test]
fn need_reply_is_set_and_acks_are_checked() {
    let fake = Fake::with_protocol(protocol::REPLY_ACK);
    let (mut frontend, handle) = negotiated(fake, protocol::REPLY_ACK);
    frontend.set_owner().unwrap();
    frontend.set_vring_num(0, 128).unwrap();
    assert_eq!(frontend.get_vring_base(0).unwrap(), 42);
    let log = finish(frontend, handle);
    assert_eq!(log[2].request, request::SET_PROTOCOL_FEATURES);
    assert_eq!(log[2].flags, FLAG_VERSION, "SET_PROTOCOL_FEATURES itself is not acked");
    assert_eq!(log[3].flags, FLAG_VERSION | FLAG_NEED_REPLY);
    assert_eq!(log[4].flags, FLAG_VERSION | FLAG_NEED_REPLY);
    assert_eq!(log[5].flags, FLAG_VERSION, "requests with a reply do not ask for an ack");

    let fake = Fake { ack_status: 5, ..Fake::with_protocol(protocol::REPLY_ACK) };
    let (mut frontend, handle) = negotiated(fake, protocol::REPLY_ACK);
    assert!(matches!(
        frontend.set_owner(),
        Err(Error::BackendFailed { request: request::SET_OWNER, status: 5 })
    ));
    finish(frontend, handle);
}

#[test]
fn get_and_set_config() {
    let fake = Fake::with_protocol(protocol::CONFIG | protocol::REPLY_ACK);
    let (mut frontend, handle) = negotiated(fake, u64::MAX);
    assert_eq!(frontend.get_config(4, 8, 0).unwrap(), (4u8..12).collect::<Vec<_>>());
    frontend.set_config(2, 1, &[0xaa, 0xbb]).unwrap();
    assert!(matches!(frontend.get_config(0, 0, 0), Err(Error::InvalidArgument(_))));
    assert!(matches!(frontend.get_config(0, 257, 0), Err(Error::InvalidArgument(_))));
    let log = finish(frontend, handle);
    let get = log.iter().find(|m| m.request == request::GET_CONFIG).unwrap();
    assert_eq!(get.payload.len(), 12 + 8);
    assert_eq!((get.u32(0), get.u32(4), get.u32(8)), (4, 8, 0));
    let set = log.iter().find(|m| m.request == request::SET_CONFIG).unwrap();
    assert_eq!((set.u32(0), set.u32(4), set.u32(8)), (2, 2, 1));
    assert_eq!(&set.payload[12..], &[0xaa, 0xbb]);
    assert_eq!(set.flags, FLAG_VERSION | FLAG_NEED_REPLY);

    // An empty GET_CONFIG reply is how a backend says it failed.
    let mut fake = Fake::with_protocol(protocol::CONFIG);
    fake.hook = Some(Box::new(|s, m| {
        if m.request == request::GET_CONFIG {
            reply(s, m.request, &[]);
            Hooked::Handled
        } else {
            Hooked::Pass
        }
    }));
    let (mut frontend, handle) = negotiated(fake, protocol::CONFIG);
    assert!(matches!(frontend.get_config(0, 4, 0), Err(Error::BackendFailed { .. })));
    finish(frontend, handle);
}

#[test]
fn log_base_and_fd() {
    let (mut frontend, handle) = negotiated(Fake::default(), protocol::LOG_SHMFD);
    let log_file = memory_file(0x2000);
    let (event, _peer) = UnixStream::pair().unwrap();
    // 256 MiB of guest memory needs one bit per 4 KiB page, as vhost-user-test.c checks.
    let size = (256 * 1024 * 1024) / (0x1000 * 8);
    frontend
        .set_log_base(0, Some(LogRegion { fd: log_file.as_fd(), size, offset: 0x100 }))
        .unwrap();
    frontend.set_log_fd(event.as_fd()).unwrap();
    frontend.set_log_base(0x1234, None).unwrap();
    let log = finish(frontend, handle);
    let bases: Vec<_> = log.iter().filter(|m| m.request == request::SET_LOG_BASE).collect();
    assert_eq!((bases[0].u64(0), bases[0].u64(8), bases[0].fds.len()), (size, 0x100, 1));
    assert_eq!((bases[1].u64(0), bases[1].fds.len()), (0x1234, 0));
    let fd = log.iter().find(|m| m.request == request::SET_LOG_FD).unwrap();
    assert_eq!((fd.payload.len(), fd.fds.len()), (0, 1));

    let (mut frontend, handle) = negotiated(Fake::default(), 0);
    let region = LogRegion { fd: log_file.as_fd(), size: 1, offset: 0 };
    assert!(matches!(frontend.set_log_base(0, Some(region)), Err(Error::NotNegotiated(_))));
    finish(frontend, handle);
}

#[test]
fn status_and_reset() {
    let offered = protocol::STATUS | protocol::RESET_DEVICE;
    let (mut frontend, handle) = negotiated(Fake::with_protocol(offered), protocol::STATUS);
    frontend.set_status(0x0f).unwrap();
    assert_eq!(frontend.get_status().unwrap(), 0x0f);
    frontend.reset().unwrap();
    let log = finish(frontend, handle);
    assert_eq!(log.last().unwrap().request, request::RESET_OWNER);

    let (mut frontend, handle) = negotiated(Fake::with_protocol(offered), offered);
    frontend.reset().unwrap();
    let log = finish(frontend, handle);
    assert_eq!(log.last().unwrap().request, request::RESET_DEVICE);
}

fn failing(hook: Hook) -> Frontend {
    let fake = Fake { hook: Some(hook), ..Fake::default() };
    let (frontend, handle) = fake.start();
    // The fake has either hung up or answered; nothing more is coming from it.
    drop(handle);
    frontend
}

#[test]
fn error_paths() {
    let mut f = failing(Box::new(|_, _| Hooked::HangUp));
    assert!(matches!(f.get_features(), Err(Error::Disconnected)));

    let mut f = failing(Box::new(|s, m| {
        // A header that promises 8 bytes, then 4, then the socket closes.
        let mut bytes = Vec::new();
        for w in [m.request, FLAG_VERSION | FLAG_REPLY, 8] {
            bytes.extend_from_slice(&w.to_ne_bytes());
        }
        bytes.extend_from_slice(&[0, 0, 0, 0]);
        (&*s).write_all(&bytes).unwrap();
        Hooked::HangUp
    }));
    assert!(matches!(f.get_features(), Err(Error::ShortMessage { expected: 20, got: 16 })));

    let mut f = failing(Box::new(|s, _| {
        (&*s).write_all(&[1, 0, 0, 0, 5]).unwrap();
        Hooked::HangUp
    }));
    assert!(matches!(f.get_features(), Err(Error::ShortMessage { expected: 12, got: 5 })));

    let mut f = failing(Box::new(|s, _| {
        reply(s, request::GET_PROTOCOL_FEATURES, &0u64.to_ne_bytes());
        Hooked::Handled
    }));
    assert!(matches!(
        f.get_features(),
        Err(Error::UnexpectedReply { expected: request::GET_FEATURES, got: 15 })
    ));

    let mut f = failing(Box::new(|s, m| {
        send_raw(s, m.request, FLAG_VERSION, &0u64.to_ne_bytes());
        Hooked::Handled
    }));
    assert!(matches!(f.get_features(), Err(Error::BadFlags { .. })));

    let mut f = failing(Box::new(|s, m| {
        send_raw(s, m.request, 2 | FLAG_REPLY, &0u64.to_ne_bytes());
        Hooked::Handled
    }));
    assert!(matches!(f.get_features(), Err(Error::BadVersion { flags: 6 })));

    let mut f = failing(Box::new(|s, m| {
        reply(s, m.request, &0u32.to_ne_bytes());
        Hooked::Handled
    }));
    assert!(matches!(f.get_features(), Err(Error::BadPayloadSize { expected: 8, got: 4, .. })));

    let mut f = failing(Box::new(|s, m| {
        let mut bytes = Vec::new();
        for w in [m.request, FLAG_VERSION | FLAG_REPLY, 1 << 20] {
            bytes.extend_from_slice(&w.to_ne_bytes());
        }
        (&*s).write_all(&bytes).unwrap();
        Hooked::Handled
    }));
    assert!(matches!(f.get_features(), Err(Error::PayloadTooLarge(0x10_0000))));
}

#[test]
fn backend_channel_config_change() {
    let (tx, rx) = mpsc::channel();
    let offered = protocol::BACKEND_REQ | protocol::REPLY_ACK | protocol::CONFIG;
    let fake = Fake { backend_fd: Some(tx), ..Fake::with_protocol(offered) };
    let (mut frontend, handle) = negotiated(fake, offered);

    let seen = Arc::new(Mutex::new(Vec::new()));
    let record = Arc::clone(&seen);
    let mut channel = frontend
        .set_backend_req_fd(Box::new(move |req| {
            let (name, status) = match req {
                BackendRequest::ConfigChange => ("config", 0),
                BackendRequest::VringCall { index } => {
                    assert_eq!(index, 3);
                    ("call", 0)
                }
                _ => ("other", 1),
            };
            record.lock().unwrap().push(name);
            status
        }))
        .unwrap();
    let backend_end = UnixStream::from(rx.recv().unwrap());

    // The backend announces a config change and wants an ack.
    send_raw(&backend_end, 2, FLAG_VERSION | FLAG_NEED_REPLY, &[]);
    assert!(channel.handle_one().unwrap());
    let ack = recv(&backend_end).unwrap();
    assert_eq!((ack.request, ack.flags, ack.u64(0)), (2, FLAG_VERSION | FLAG_REPLY, 0));
    // The VMM's answer to a config change is to read the config space again.
    assert_eq!(frontend.get_config(0, 4, 0).unwrap(), [0, 1, 2, 3]);

    // Without need_reply there is no answer; an unknown request gets a failure status.
    let mut state = 3u32.to_ne_bytes().to_vec();
    state.extend_from_slice(&0u32.to_ne_bytes());
    send_raw(&backend_end, 4, FLAG_VERSION, &state);
    send_raw(&backend_end, 99, FLAG_VERSION | FLAG_NEED_REPLY, &[]);
    assert!(channel.handle_one().unwrap());
    assert!(channel.handle_one().unwrap());
    let nak = recv(&backend_end).unwrap();
    assert_eq!((nak.request, nak.u64(0)), (99, 1));

    // A backend request must not carry the reply flag.
    send_raw(&backend_end, 2, FLAG_VERSION | FLAG_REPLY, &[]);
    assert!(matches!(channel.handle_one(), Err(Error::BadFlags { .. })));

    drop(backend_end);
    assert!(!channel.handle_one().unwrap());
    assert_eq!(*seen.lock().unwrap(), ["config", "call", "other"]);
    let log = finish(frontend, handle);
    let set = log.iter().find(|m| m.request == request::SET_BACKEND_REQ_FD).unwrap();
    assert_eq!(set.flags, FLAG_VERSION | FLAG_NEED_REPLY);
}

#[test]
fn frontend_works_through_the_common_trait() {
    let (frontend, handle) = negotiated(Fake::default(), 0);
    let mut boxed: Box<dyn VhostBackend> = Box::new(frontend);
    let dev = boxed.as_mut();
    dev.set_owner().unwrap();
    assert_ne!(dev.get_features().unwrap() & VIRTIO_F_VERSION_1, 0);
    dev.set_features(VIRTIO_F_VERSION_1).unwrap();
    dev.set_vring_num(0, 64).unwrap();
    assert_eq!(dev.get_vring_base(0).unwrap(), 42);
    dev.reset_owner().unwrap();
    drop(boxed);
    let log = handle.join().unwrap();
    assert_eq!(log.last().unwrap().request, request::RESET_OWNER);
}
