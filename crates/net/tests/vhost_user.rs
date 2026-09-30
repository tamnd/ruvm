// SPDX-License-Identifier: GPL-2.0-or-later

//! `-netdev vhost-user` against an in-process fake backend on the other end of a socket.
//!
//! The fake decodes the wire format by hand, like the one in crates/vhost/tests/vhost_user.rs,
//! answers the requests that have replies, acks `need_reply` requests and records every
//! request it was sent.

#![cfg(unix)]

use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use ruvm_net::{Net, NetClient, VhostUserChardev};
use ruvm_vhost::VHOST_USER_F_PROTOCOL_FEATURES;
use ruvm_vhost::user::message::{protocol, request};

const VIRTIO_F_VERSION_1: u64 = 1 << 32;
const VIRTIO_NET_F_MRG_RXBUF: u64 = 1 << 15;
const FLAG_VERSION: u32 = 1;
const FLAG_REPLY: u32 = 1 << 2;
const FLAG_NEED_REPLY: u32 = 1 << 3;

/// What the fake offers.
#[derive(Clone, Copy)]
struct Offer {
    features: u64,
    protocol: u64,
    queues: u64,
}

impl Default for Offer {
    fn default() -> Self {
        Offer {
            features: VIRTIO_F_VERSION_1 | VIRTIO_NET_F_MRG_RXBUF | VHOST_USER_F_PROTOCOL_FEATURES,
            protocol: protocol::MQ | protocol::REPLY_ACK | protocol::PAGEFAULT,
            queues: 4,
        }
    }
}

/// Every request the fake got, with its payload.
type Seen = Arc<Mutex<Vec<(u32, Vec<u8>)>>>;

/// The fake backend: a thread serving one connection, and what it was asked.
struct Fake {
    seen: Seen,
    thread: Option<JoinHandle<()>>,
}

fn reply(s: &mut UnixStream, req: u32, payload: &[u8]) {
    let mut b = Vec::new();
    b.extend_from_slice(&req.to_ne_bytes());
    b.extend_from_slice(&(FLAG_VERSION | FLAG_REPLY).to_ne_bytes());
    b.extend_from_slice(&(payload.len() as u32).to_ne_bytes());
    b.extend_from_slice(payload);
    s.write_all(&b).unwrap();
}

impl Fake {
    fn serve(mut s: UnixStream, offer: Offer) -> Fake {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = seen.clone();
        let thread = std::thread::spawn(move || {
            loop {
                let mut hdr = [0u8; 12];
                if s.read_exact(&mut hdr).is_err() {
                    return;
                }
                let word = |i: usize| u32::from_ne_bytes(hdr[i..i + 4].try_into().unwrap());
                let (req, flags) = (word(0), word(4));
                let mut payload = vec![0u8; word(8) as usize];
                if s.read_exact(&mut payload).is_err() {
                    return;
                }
                log.lock().unwrap().push((req, payload.clone()));
                match req {
                    request::GET_FEATURES => reply(&mut s, req, &offer.features.to_ne_bytes()),
                    request::GET_PROTOCOL_FEATURES => {
                        reply(&mut s, req, &offer.protocol.to_ne_bytes());
                    }
                    request::GET_QUEUE_NUM => reply(&mut s, req, &offer.queues.to_ne_bytes()),
                    _ if flags & FLAG_NEED_REPLY != 0 => reply(&mut s, req, &0u64.to_ne_bytes()),
                    _ => {}
                }
            }
        });
        Fake { seen, thread: Some(thread) }
    }

    fn requests(&self) -> Vec<u32> {
        self.seen.lock().unwrap().iter().map(|(r, _)| r.to_owned()).collect()
    }

    fn payload(&self, req: u32) -> Vec<u8> {
        self.seen.lock().unwrap().iter().find(|(r, _)| *r == req).unwrap().1.clone()
    }
}

impl Drop for Fake {
    fn drop(&mut self) {
        // The thread ends when the frontend side closes.
        if let Some(t) = self.thread.take() {
            if t.is_finished() {
                t.join().unwrap();
            }
        }
    }
}

/// A `Net` whose chardev lookup knows `chr0`, connected to a fake with `offer`.
fn with_backend(offer: Offer, reconnectable: bool, fd_pass: bool) -> (Net, Fake) {
    let (ours, theirs) = UnixStream::pair().unwrap();
    let fake = Fake::serve(theirs, offer);
    let mut slot = Some(VhostUserChardev::new("chr0", ours, reconnectable, fd_pass));
    let mut net = Net::new();
    net.set_chardev_resolver(Some(Box::new(
        move |label| {
            if label == "chr0" { slot.take() } else { None }
        },
    )));
    (net, fake)
}

fn clients_named(net: &Net, name: &str) -> Vec<Arc<NetClient>> {
    net.clients().iter().filter(|c| c.name() == name).cloned().collect()
}

fn err(r: ruvm_base::Result<()>) -> String {
    r.unwrap_err().message().to_string()
}

#[test]
fn handshake_and_queues() {
    let (mut net, fake) = with_backend(Offer::default(), true, true);
    net.netdev_add_opts("vhost-user,id=vu0,chardev=chr0,queues=2").unwrap();
    let queues = clients_named(&net, "vu0");
    assert_eq!(queues.len(), 2);
    for (i, nc) in queues.iter().enumerate() {
        assert_eq!(nc.model(), "vhost_user");
        assert_eq!(nc.queue_index(), i as u32);
        assert_eq!(nc.info_str(), format!("vhost-user{i} to chr0"));
        assert!(nc.has_vnet_hdr());
        assert!(nc.has_ufo());
    }
    assert!(queues[0].is_netdev());

    let vu = queues[0].vhost_user().expect("a vhost-user client");
    assert!(Arc::ptr_eq(&vu, &queues[1].vhost_user().unwrap()));
    assert_eq!(vu.features(), Offer::default().features);
    assert_eq!(vu.protocol_features(), protocol::MQ | protocol::REPLY_ACK);
    assert_eq!(vu.max_queues(), 4);
    assert_eq!(vu.queues(), 2);
    assert_eq!(vu.chardev(), "chr0");
    assert!(vu.frontend().lock().is_ok());

    assert_eq!(
        fake.requests(),
        [
            request::GET_FEATURES,
            request::GET_PROTOCOL_FEATURES,
            request::SET_PROTOCOL_FEATURES,
            request::GET_QUEUE_NUM,
            request::SET_OWNER,
        ]
    );
    let agreed = fake.payload(request::SET_PROTOCOL_FEATURES);
    assert_eq!(u64::from_ne_bytes(agreed.try_into().unwrap()), protocol::MQ | protocol::REPLY_ACK);

    // Frames are accepted and dropped while the rings are not in use.
    assert_eq!(queues[0].receive_packet(&[0u8; 64]), 64);

    net.netdev_del("vu0").unwrap();
    assert!(clients_named(&net, "vu0").is_empty());
}

#[test]
fn one_queue_without_mq() {
    let offer = Offer { protocol: protocol::LOG_SHMFD, ..Offer::default() };
    let (mut net, fake) = with_backend(offer, true, true);
    net.netdev_add_opts("vhost-user,id=vu0,chardev=chr0").unwrap();
    let vu = clients_named(&net, "vu0")[0].vhost_user().unwrap();
    assert_eq!(vu.max_queues(), 1);
    assert_eq!(vu.queues(), 1);
    assert_eq!(vu.protocol_features(), protocol::LOG_SHMFD);
    assert!(!fake.requests().contains(&request::GET_QUEUE_NUM));
}

#[test]
fn no_protocol_features() {
    let offer = Offer { features: VIRTIO_F_VERSION_1, ..Offer::default() };
    let (mut net, fake) = with_backend(offer, true, true);
    net.netdev_add_opts("vhost-user,id=vu0,chardev=chr0").unwrap();
    let vu = clients_named(&net, "vu0")[0].vhost_user().unwrap();
    assert_eq!(vu.protocol_features(), 0);
    assert_eq!(fake.requests(), [request::GET_FEATURES, request::SET_OWNER]);
}

#[test]
fn more_queues_than_the_backend_has() {
    let (mut net, _fake) = with_backend(Offer::default(), true, true);
    let e = err(net.netdev_add_opts("vhost-user,id=vu0,chardev=chr0,queues=8"));
    assert_eq!(e, "Device 'vhost-user' could not be initialized");
    assert!(clients_named(&net, "vu0").is_empty());

    let offer = Offer { protocol: 0, ..Offer::default() };
    let (mut net, _fake) = with_backend(offer, true, true);
    let e = err(net.netdev_add_opts("vhost-user,id=vu0,chardev=chr0,queues=2"));
    assert_eq!(e, "Device 'vhost-user' could not be initialized");
}

#[test]
fn backend_hangs_up() {
    let (ours, theirs) = UnixStream::pair().unwrap();
    drop(theirs);
    let mut slot = Some(VhostUserChardev::new("chr0", ours, true, true));
    let mut net = Net::new();
    net.set_chardev_resolver(Some(Box::new(move |_| slot.take())));
    let e = err(net.netdev_add_opts("vhost-user,id=vu0,chardev=chr0"));
    assert_eq!(e, "Device 'vhost-user' could not be initialized");
}

#[test]
fn chardev_errors() {
    let mut net = Net::new();
    let e = err(net.netdev_add_opts("vhost-user,id=vu0,chardev=nope"));
    assert_eq!(e, "chardev \"nope\" not found");

    let (mut net, _fake) = with_backend(Offer::default(), true, true);
    let e = err(net.netdev_add_opts("vhost-user,id=vu0,chardev=other"));
    assert_eq!(e, "chardev \"other\" not found");

    let (mut net, _fake) = with_backend(Offer::default(), false, true);
    let e = err(net.netdev_add_opts("vhost-user,id=vu0,chardev=chr0"));
    assert_eq!(e, "chardev \"chr0\" is not reconnectable");

    let (mut net, _fake) = with_backend(Offer::default(), true, false);
    let e = err(net.netdev_add_opts("vhost-user,id=vu0,chardev=chr0"));
    assert_eq!(e, "chardev \"chr0\" does not support FD passing");

    let mut net = Net::new();
    let e = err(net.netdev_add_opts("vhost-user,id=vu0"));
    assert_eq!(e, "Parameter 'chardev' is missing");
}

#[test]
fn queue_range() {
    for q in ["0", "1025"] {
        let (mut net, fake) = with_backend(Offer::default(), true, true);
        let e = err(net.netdev_add_opts(&format!("vhost-user,id=vu0,chardev=chr0,queues={q}")));
        assert_eq!(e, "vhost-user number of queues must be in range [1, 1024]");
        assert!(fake.requests().is_empty());
    }
}

#[test]
fn connect_to_a_socket_path() {
    let dir = std::env::temp_dir().join(format!("ruvm-net-vu-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("vu.sock");
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).unwrap();
    let accept = std::thread::spawn(move || listener.accept().unwrap().0);
    let chr = VhostUserChardev::connect("sock0", &path).unwrap();
    assert_eq!(chr.label(), "sock0");
    let fake = Fake::serve(accept.join().unwrap(), Offer::default());
    let mut slot = Some(chr);
    let mut net = Net::new();
    net.set_chardev_resolver(Some(Box::new(move |_| slot.take())));
    net.parse_netdev("vhost-user,id=vu1,chardev=sock0").unwrap();
    net.init_clients().unwrap();
    let nc = net.find_netdev("vu1").unwrap();
    assert_eq!(nc.info_str(), "vhost-user0 to sock0");
    assert_eq!(nc.vhost_user().unwrap().chardev(), "sock0");
    assert!(fake.requests().contains(&request::SET_OWNER));
    assert!(net.info_network().contains("vu1: index=0,type=vhost-user,vhost-user0 to sock0"));
    std::fs::remove_dir_all(&dir).unwrap();
}
