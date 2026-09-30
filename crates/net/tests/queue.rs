// SPDX-License-Identifier: GPL-2.0-or-later

//! Delivery and queueing between two peers, as in net/queue.c.

mod common;

use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use common::{Recorder, frame, plain_client};
use ruvm_net::{
    MacAddr, Net, NetClient, Nic, NicConf, SentCb, VNET_HDR_LEN, VNET_HDR_MRG_RXBUF_LEN,
};

/// A backend client and a NIC joined to it. Frames the backend sends reach `rec`.
fn pair(net: &mut Net) -> (Arc<NetClient>, Arc<Nic>, Arc<Recorder>, Arc<Recorder>) {
    let (backend, brec) = plain_client(net, "be");
    let rec = Recorder::new();
    let r = rec.clone();
    let conf = NicConf {
        macaddr: MacAddr([0x52, 0x54, 0, 0x12, 0x34, 0x56]),
        peers: vec![backend.clone()],
    };
    let nic = net.new_nic(&conf, "test-nic", Some("nic"), move |_, _| r.clone());
    (backend, nic, rec, brec)
}

/// A sent callback that stores what it was given and counts calls.
fn cb_probe() -> (Arc<(AtomicUsize, AtomicIsize)>, impl Fn() -> SentCb) {
    let st = Arc::new((AtomicUsize::new(0), AtomicIsize::new(-1)));
    let s = st.clone();
    let make = move || -> SentCb {
        let s = s.clone();
        Box::new(move |ret| {
            s.0.fetch_add(1, Ordering::SeqCst);
            s.1.store(ret, Ordering::SeqCst);
        })
    };
    (st, make)
}

#[test]
fn delivers_directly() {
    let mut net = Net::new();
    let (be, nic, rec, _) = pair(&mut net);
    assert!(Arc::ptr_eq(&be.peer().unwrap(), nic.queue()));
    let f = frame(1, 60);
    assert_eq!(be.send_packet(&f), 60);
    assert_eq!(rec.take(), vec![f]);
    assert!(nic.queue().incoming_queue().is_empty());
}

#[test]
fn sendv_joins_iovecs() {
    let mut net = Net::new();
    let (be, _nic, rec, _) = pair(&mut net);
    assert_eq!(be.sendv_packet(&[b"abc", b"def"]), 6);
    assert_eq!(rec.take(), vec![b"abcdef".to_vec()]);
}

#[test]
fn queues_while_peer_cannot_receive_then_flushes() {
    let mut net = Net::new();
    let (be, nic, rec, _) = pair(&mut net);
    let (st, cb) = cb_probe();
    rec.closed.store(true, Ordering::SeqCst);
    assert_eq!(be.send_packet_async(&frame(1, 64), Some(cb())), 0);
    assert_eq!(be.send_packet_async(&frame(2, 64), Some(cb())), 0);
    assert_eq!(nic.queue().incoming_queue().len(), 2);
    assert_eq!(rec.count(), 0);

    rec.closed.store(false, Ordering::SeqCst);
    nic.queue().flush_queued_packets();
    assert_eq!(rec.take(), vec![frame(1, 64), frame(2, 64)]);
    assert_eq!(st.0.load(Ordering::SeqCst), 2);
    assert_eq!(st.1.load(Ordering::SeqCst), 64);
    assert!(nic.queue().incoming_queue().is_empty());
}

#[test]
fn full_receiver_disables_receive_until_flush() {
    let mut net = Net::new();
    let (be, nic, rec, _) = pair(&mut net);
    rec.full.store(true, Ordering::SeqCst);
    assert_eq!(be.send_packet(&frame(3, 70)), 0);
    assert!(nic.queue().receive_disabled());
    assert_eq!(nic.queue().incoming_queue().len(), 1);
    // While disabled, the sender sees a peer that cannot take anything.
    assert!(!be.can_send_packet());
    assert_eq!(be.send_packet(&frame(4, 70)), 0);
    assert_eq!(nic.queue().incoming_queue().len(), 2);

    rec.full.store(false, Ordering::SeqCst);
    nic.queue().flush_queued_packets();
    assert!(!nic.queue().receive_disabled());
    assert_eq!(rec.take(), vec![frame(3, 70), frame(4, 70)]);
}

#[test]
fn purge_tells_the_sender() {
    let mut net = Net::new();
    let (be, nic, rec, _) = pair(&mut net);
    let (st, cb) = cb_probe();
    rec.closed.store(true, Ordering::SeqCst);
    be.send_packet_async(&frame(5, 60), Some(cb()));
    be.send_packet_async(&frame(6, 60), Some(cb()));
    be.purge_queued_packets();
    assert_eq!(st.0.load(Ordering::SeqCst), 2);
    assert_eq!(st.1.load(Ordering::SeqCst), 0);
    assert!(nic.queue().incoming_queue().is_empty());
    rec.closed.store(false, Ordering::SeqCst);
    nic.queue().flush_queued_packets();
    assert_eq!(rec.count(), 0);
}

#[test]
fn link_down_drops_but_reports_sent() {
    let mut net = Net::new();
    let (be, nic, rec, brec) = pair(&mut net);
    net.set_link("be", false).unwrap();
    assert!(be.link_down());
    // The NIC is the peer of a backend, so its link goes down too.
    assert!(nic.queue().link_down());
    assert_eq!(brec.link_changes.load(Ordering::SeqCst), 1);
    assert_eq!(rec.link_changes.load(Ordering::SeqCst), 1);
    assert_eq!(be.send_packet(&frame(7, 80)), 80);
    assert_eq!(rec.count(), 0);

    net.set_link("be", true).unwrap();
    assert_eq!(be.send_packet(&frame(7, 80)), 80);
    assert_eq!(rec.take(), vec![frame(7, 80)]);

    let e = net.set_link("nope", true).unwrap_err();
    assert_eq!(e.message(), "Device 'nope' not found");
}

#[test]
fn stopped_vm_queues_until_running() {
    let mut net = Net::new();
    let (be, nic, rec, _) = pair(&mut net);
    net.vm_state_change(false);
    assert!(!be.can_send_packet());
    assert_eq!(be.send_packet(&frame(8, 60)), 0);
    assert_eq!(nic.queue().incoming_queue().len(), 1);
    net.vm_state_change(true);
    assert_eq!(rec.take(), vec![frame(8, 60)]);
}

#[test]
fn stopping_vm_purges_what_cannot_be_delivered() {
    let mut net = Net::new();
    let (be, nic, rec, _) = pair(&mut net);
    let (st, cb) = cb_probe();
    rec.closed.store(true, Ordering::SeqCst);
    rec.full.store(true, Ordering::SeqCst);
    be.send_packet_async(&frame(9, 60), Some(cb()));
    rec.closed.store(false, Ordering::SeqCst);
    net.vm_state_change(false);
    assert!(nic.queue().incoming_queue().is_empty());
    assert_eq!(st.0.load(Ordering::SeqCst), 1);
    assert_eq!(st.1.load(Ordering::SeqCst), 0);
}

#[test]
fn full_queue_drops_packets_without_callback() {
    let mut net = Net::new();
    let (be, nic, rec, _) = pair(&mut net);
    rec.closed.store(true, Ordering::SeqCst);
    nic.queue().incoming_queue().set_maxlen(2);
    for i in 0..4 {
        be.send_packet(&frame(i, 60));
    }
    assert_eq!(nic.queue().incoming_queue().len(), 2);
    // With a callback the packet is kept anyway, since someone waits for it.
    let (_st, cb) = cb_probe();
    be.send_packet_async(&frame(10, 60), Some(cb()));
    assert_eq!(nic.queue().incoming_queue().len(), 3);
    rec.closed.store(false, Ordering::SeqCst);
    nic.queue().flush_queued_packets();
    assert_eq!(rec.take(), vec![frame(0, 60), frame(1, 60), frame(10, 60)]);
}

#[test]
fn no_peer_swallows_packets() {
    let mut net = Net::new();
    let (nc, _) = plain_client(&mut net, "lonely");
    assert!(nc.peer().is_none());
    assert!(nc.can_send_packet());
    assert_eq!(nc.send_packet(&frame(1, 42)), 42);
}

#[test]
fn vnet_hdr_len_needs_the_backend() {
    let mut net = Net::new();
    let (be, _nic, _rec, brec) = pair(&mut net);
    assert_eq!(be.vnet_hdr_len(), 0);
    be.set_vnet_hdr_len(VNET_HDR_MRG_RXBUF_LEN);
    assert_eq!(be.vnet_hdr_len(), VNET_HDR_MRG_RXBUF_LEN);
    brec.no_vnet_hdr.store(true, Ordering::SeqCst);
    be.set_vnet_hdr_len(VNET_HDR_LEN);
    assert_eq!(be.vnet_hdr_len(), VNET_HDR_MRG_RXBUF_LEN);
}

#[test]
fn raw_packets_get_a_zero_header() {
    let mut net = Net::new();
    let (be, nic, rec, _) = pair(&mut net);
    // The receiver of a raw packet is the one with the header.
    nic.queue().set_vnet_hdr_len(VNET_HDR_LEN);
    assert_eq!(be.send_packet_raw(b"xyz"), 3 + VNET_HDR_LEN as isize);
    let mut want = vec![0u8; VNET_HDR_LEN];
    want.extend_from_slice(b"xyz");
    assert_eq!(rec.take(), vec![want]);
    // Normal packets go as they are.
    be.send_packet(b"xyz");
    assert_eq!(rec.take(), vec![b"xyz".to_vec()]);
}

#[test]
fn receive_packet_pads_short_frames() {
    let mut net = Net::new();
    let (_be, nic, rec, _) = pair(&mut net);
    assert_eq!(nic.queue().receive_packet(b"short"), 60);
    let got = rec.take();
    assert_eq!(got[0].len(), 60);
    assert_eq!(&got[0][..5], b"short");
    nic.queue().set_do_not_pad(true);
    assert_eq!(nic.queue().receive_packet(b"short"), 5);
}

#[test]
fn callbacks_can_send_again() {
    // A callback that sends from inside the flush must not deadlock.
    let mut net = Net::new();
    let (be, nic, rec, _) = pair(&mut net);
    rec.closed.store(true, Ordering::SeqCst);
    let slot: Arc<Mutex<Option<Arc<NetClient>>>> = Arc::new(Mutex::new(Some(be.clone())));
    let s = slot.clone();
    be.send_packet_async(
        &frame(1, 60),
        Some(Box::new(move |_| {
            if let Some(be) = s.lock().unwrap().take() {
                be.send_packet(&frame(2, 60));
            }
        })),
    );
    rec.closed.store(false, Ordering::SeqCst);
    nic.queue().flush_queued_packets();
    assert_eq!(rec.take(), vec![frame(1, 60), frame(2, 60)]);
}

#[test]
fn deleting_a_backend_detaches_the_nic() {
    let mut net = Net::new();
    let (be, nic, _rec, _) = pair(&mut net);
    net.del_client(&be);
    assert!(nic.peer_deleted() || nic.queue().peer().is_none());
    assert!(net.find_netdev("be").is_none());
}
