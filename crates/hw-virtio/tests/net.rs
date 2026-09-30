// SPDX-License-Identifier: GPL-2.0-or-later

//! virtio-net through virtio-mmio: the checks from `tests/qtest/virtio-net-test.c` with a
//! loopback peer instead of a socket, plus the receive filter, the control queue and
//! multiqueue.

mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use common::{Buf, Guest, RAM_SIZE};
use ruvm_hw_virtio::mmio::{VIRTIO_MMIO_CONFIG, VIRTIO_MMIO_DEVICE_ID};
use ruvm_hw_virtio::net::*;
use ruvm_hw_virtio::virtio::*;
use ruvm_hw_virtio::{NetPeer, VirtioDeviceClass};

type Sent = Arc<Mutex<Vec<(u16, Option<VirtioNetHdr>, Vec<u8>)>>>;

/// Collects what the guest sends, the other end of the socket in the qtest. It cannot hand
/// frames straight back to the device, since the device is busy sending them.
#[derive(Debug, Clone, Default)]
struct Loopback {
    sent: Sent,
    ready: Arc<Mutex<Vec<u16>>>,
    offloads: Arc<Mutex<Vec<u64>>>,
    busy: Arc<AtomicBool>,
    vnet_hdr: bool,
}

impl Loopback {
    fn frames(&self) -> Vec<Vec<u8>> {
        self.sent.lock().unwrap().iter().map(|(_, _, f)| f.clone()).collect()
    }
}

impl NetPeer for Loopback {
    fn has_vnet_hdr(&self) -> bool {
        self.vnet_hdr
    }

    fn can_receive(&self, _pair: u16) -> bool {
        !self.busy.load(Ordering::SeqCst)
    }

    fn send(&mut self, pair: u16, hdr: Option<&VirtioNetHdr>, frame: &[u8]) {
        self.sent.lock().unwrap().push((pair, hdr.copied(), frame.to_vec()));
    }

    fn set_offloads(&mut self, offloads: u64) {
        self.offloads.lock().unwrap().push(offloads);
    }

    fn rx_ready(&mut self, pair: u16) {
        self.ready.lock().unwrap().push(pair);
    }
}

const OUR_MAC: [u8; 6] = [0x52, 0x54, 0x00, 0x12, 0x34, 0x56];
const OTHER_MAC: [u8; 6] = [0x52, 0x54, 0x00, 0xaa, 0xbb, 0xcc];
const BCAST: [u8; 6] = [0xff; 6];
const MCAST: [u8; 6] = [0x01, 0x00, 0x5e, 0x00, 0x00, 0x01];

fn net(conf: VirtioNetConf, peer: &Loopback) -> Option<Box<dyn VirtioDeviceClass>> {
    Some(Box::new(VirtioNet::new(conf, Some(Box::new(peer.clone())))))
}

fn features() -> u64 {
    !feature(VIRTIO_RING_F_EVENT_IDX)
}

fn frame(dst: [u8; 6], payload: &[u8]) -> Vec<u8> {
    let mut f = dst.to_vec();
    f.extend_from_slice(&OTHER_MAC);
    f.extend_from_slice(&[0x08, 0x00]);
    f.extend_from_slice(payload);
    f
}

fn receive(g: &Guest, pair: u16, f: &[u8]) -> RxOutcome {
    g.mmio.with_device(|vdev, d: &mut VirtioNet| d.receive_on(vdev, pair, None, f)).unwrap()
}

fn dev<R>(g: &Guest, f: impl FnOnce(&mut VirtIODevice, &mut VirtioNet) -> R) -> R {
    g.mmio.with_device(f).unwrap()
}

/// A negotiated device with rx0 and tx0 set up and DRIVER_OK.
fn start(
    legacy: bool,
    conf: VirtioNetConf,
    peer: &Loopback,
    wanted: u64,
) -> (Guest, common::SplitRing, common::SplitRing) {
    let mut g = Guest::new(legacy, net(conf, peer));
    g.negotiate(wanted);
    let rx = g.setup_queue(0, 0);
    let tx = g.setup_queue(1, 0);
    g.driver_ok();
    (g, rx, tx)
}

/// Sends one control command and returns the ack.
fn ctrl(g: &mut Guest, ring: &mut common::SplitRing, class: u8, cmd: u8, data: &[u8]) -> u8 {
    let mut out = vec![class, cmd];
    out.extend_from_slice(data);
    let o = g.alloc(out.len() as u64, 8);
    g.write_mem(o, &out);
    let i = g.alloc(1, 8);
    g.write_mem(i, &[0xaa]);
    let head = ring.submit(g, &[Buf::out(o, out.len() as u32), Buf::inp(i, 1)]);
    g.kick(ring);
    assert_eq!(ring.get_used(g), Some((u32::from(head), 1)));
    g.read_mem(i, 1)[0]
}

fn mac_table(uni: &[[u8; 6]], multi: &[[u8; 6]]) -> Vec<u8> {
    let mut d = (uni.len() as u32).to_le_bytes().to_vec();
    uni.iter().for_each(|m| d.extend_from_slice(m));
    d.extend_from_slice(&(multi.len() as u32).to_le_bytes());
    multi.iter().for_each(|m| d.extend_from_slice(m));
    d
}

#[test]
fn identity_features_and_config() {
    for legacy in [true, false] {
        let peer = Loopback::default();
        let g = Guest::new(legacy, net(VirtioNetConf::default(), &peer));
        assert_eq!(g.readl(VIRTIO_MMIO_DEVICE_ID), 1);
        let f = g.device_features();
        for bit in [
            VIRTIO_NET_F_MAC,
            VIRTIO_NET_F_MRG_RXBUF,
            VIRTIO_NET_F_STATUS,
            VIRTIO_NET_F_CTRL_VQ,
            VIRTIO_NET_F_CTRL_RX,
            VIRTIO_NET_F_CTRL_VLAN,
            VIRTIO_NET_F_CTRL_RX_EXTRA,
            VIRTIO_NET_F_GUEST_ANNOUNCE,
            VIRTIO_NET_F_CTRL_MAC_ADDR,
            VIRTIO_NET_F_CTRL_GUEST_OFFLOADS,
        ] {
            assert!(has_feature(f, bit), "bit {bit}");
        }
        // Without a vnet header peer there are no offloads.
        for bit in [
            VIRTIO_NET_F_CSUM,
            VIRTIO_NET_F_GUEST_CSUM,
            VIRTIO_NET_F_HOST_TSO4,
            VIRTIO_NET_F_GUEST_UFO,
            VIRTIO_NET_F_HOST_USO,
            VIRTIO_NET_F_MQ,
            VIRTIO_NET_F_RSS,
            VIRTIO_NET_F_MTU,
            VIRTIO_NET_F_SPEED_DUPLEX,
        ] {
            assert!(!has_feature(f, bit), "bit {bit}");
        }
        // GSO is a legacy only feature.
        assert_eq!(has_feature(f, VIRTIO_NET_F_GSO), legacy);
        for (i, b) in OUR_MAC.iter().enumerate() {
            assert_eq!(g.config_readb(i as u64), *b);
        }
        assert_eq!(g.config_readw(6), VIRTIO_NET_S_LINK_UP);
    }

    // A vnet header peer gets the checksum and TSO offloads, but not UFO or USO it lacks.
    let peer = Loopback { vnet_hdr: true, ..Loopback::default() };
    let g = Guest::new(false, net(VirtioNetConf::default(), &peer));
    let f = g.device_features();
    assert!(has_feature(f, VIRTIO_NET_F_CSUM) && has_feature(f, VIRTIO_NET_F_GUEST_TSO6));
    assert!(!has_feature(f, VIRTIO_NET_F_HOST_UFO) && !has_feature(f, VIRTIO_NET_F_GUEST_USO4));
}

#[test]
fn mtu_speed_and_duplex() {
    let conf = VirtioNetConf {
        host_mtu: 1400,
        speed: 1000,
        duplex: Some("full".into()),
        ..VirtioNetConf::default()
    };
    let g = Guest::new(false, net(conf, &Loopback::default()));
    let f = g.device_features();
    assert!(has_feature(f, VIRTIO_NET_F_MTU) && has_feature(f, VIRTIO_NET_F_SPEED_DUPLEX));
    assert_eq!(g.config_readw(10), 1400);
    assert_eq!(g.config_readl(12), 1000);
    assert_eq!(g.config_readb(16), DUPLEX_FULL);

    let conf = VirtioNetConf { duplex: Some("half".into()), ..VirtioNetConf::default() };
    let g = Guest::new(false, net(conf, &Loopback::default()));
    assert_eq!(g.config_readl(12), SPEED_UNKNOWN as u32);
    assert_eq!(g.config_readb(16), DUPLEX_HALF);
}

#[test]
fn bad_properties_are_refused() {
    let cases: Vec<(VirtioNetConf, &str)> = vec![
        (
            VirtioNetConf { duplex: Some("both".into()), ..VirtioNetConf::default() },
            "'duplex' must be 'half' or 'full'",
        ),
        (
            VirtioNetConf { speed: -2, ..VirtioNetConf::default() },
            "'speed' must be between 0 and INT_MAX",
        ),
        (
            VirtioNetConf { rx_queue_size: 128, ..VirtioNetConf::default() },
            "Invalid rx_queue_size (= 128), must be a power of 2 between 256 and 1024.",
        ),
        (
            VirtioNetConf { rx_queue_size: 300, ..VirtioNetConf::default() },
            "Invalid rx_queue_size (= 300), must be a power of 2 between 256 and 1024.",
        ),
        (
            VirtioNetConf { tx_queue_size: 512, ..VirtioNetConf::default() },
            "Invalid tx_queue_size (= 512), must be a power of 2 between 256 and 256",
        ),
        (
            VirtioNetConf { queue_pairs: 512, ..VirtioNetConf::default() },
            "Invalid number of queue pairs (= 512), must be a positive integer less than 511.",
        ),
    ];
    for (conf, msg) in cases {
        let err = Guest::try_new(false, net(conf, &Loopback::default())).err().unwrap();
        assert_eq!(err.message(), msg);
    }
}

/// `rx_test()`: a frame from the peer lands in the guest buffer after the header.
#[test]
fn rx() {
    for legacy in [true, false] {
        let peer = Loopback::default();
        let (mut g, mut rx, _tx) = start(legacy, VirtioNetConf::default(), &peer, features());
        let buf = g.alloc(64, 8);
        let head = rx.submit(&g, &[Buf::inp(buf, 64)]);
        g.kick(&rx);
        assert_eq!(peer.ready.lock().unwrap().as_slice(), &[0, 0]);

        g.ack();
        assert_eq!(receive(&g, 0, b"TEST\0"), RxOutcome::Delivered);
        assert!(g.irq());
        assert_eq!(rx.get_used(&g), Some((u32::from(head), 12 + 5)));
        assert_eq!(g.read_mem(buf + 12, 5), b"TEST\0");
        // An empty header with num_buffers 1.
        assert_eq!(g.read_mem(buf, 12), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]);
    }
}

/// Legacy without mergeable buffers uses the 10 byte header and leaves the rest alone.
#[test]
fn rx_legacy_short_header() {
    let peer = Loopback::default();
    let wanted = features() & !feature(VIRTIO_NET_F_MRG_RXBUF);
    let (mut g, mut rx, _tx) = start(true, VirtioNetConf::default(), &peer, wanted);
    assert_eq!(dev(&g, |_, d| d.guest_hdr_len()), 10);
    let buf = g.alloc(64, 8);
    g.write_mem(buf, &[0xee; 64]);
    rx.submit(&g, &[Buf::inp(buf, 64)]);
    assert_eq!(receive(&g, 0, b"TEST"), RxOutcome::Delivered);
    assert_eq!(rx.get_used(&g).unwrap().1, 14);
    assert_eq!(g.read_mem(buf, 14), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b'T', b'E', b'S', b'T']);
    assert_eq!(g.read_mem(buf + 14, 1), [0xee]);
}

/// `tx_test()`: the guest's frame reaches the peer without the header.
#[test]
fn tx() {
    for legacy in [true, false] {
        let peer = Loopback::default();
        let (mut g, _rx, mut tx) = start(legacy, VirtioNetConf::default(), &peer, features());
        let req = g.alloc(64, 8);
        g.write_mem(req, &[0; 64]);
        g.write_mem(req + 12, b"TEST");
        let head = tx.submit(&g, &[Buf::out(req, 64)]);
        g.kick(&tx);
        assert_eq!(tx.get_used(&g), Some((u32::from(head), 0)));
        let sent = peer.sent.lock().unwrap().clone();
        assert_eq!(sent.len(), 1);
        let (pair, hdr, f) = &sent[0];
        assert_eq!((*pair, *hdr, f.len()), (0, None, 52));
        assert_eq!(&f[..4], b"TEST");
    }
}

/// `large_tx()`: 65 descriptors that point at one area add up to more than the net layer
/// takes. The chain is used and the frame dropped, even when it points past the end of RAM.
#[test]
fn large_tx() {
    for size in [u64::from(u32::MAX), NET_BUFSIZE as u64] {
        let peer = Loopback::default();
        let (mut g, _rx, mut tx) = start(false, VirtioNetConf::default(), &peer, features());
        let alloc_size = (size / 64) as u32;
        let req = g.alloc(0x1000, 8);
        if u64::from(alloc_size) + req > RAM_SIZE {
            assert!(size > RAM_SIZE);
        }
        let bufs: Vec<Buf> = (0..65).map(|_| Buf::out(req, alloc_size)).collect();
        let head = tx.submit(&g, &bufs);
        g.kick(&tx);
        assert_eq!(tx.get_used(&g), Some((u32::from(head), 0)));
        assert!(peer.frames().is_empty());
        assert!(!dev(&g, |vdev, _| vdev.is_broken()));
    }
}

#[test]
fn mergeable_rx_spans_several_buffers() {
    for legacy in [true, false] {
        let peer = Loopback::default();
        let (mut g, mut rx, _tx) = start(legacy, VirtioNetConf::default(), &peer, features());
        assert!(dev(&g, |_, d| d.mergeable_rx_bufs()));
        let bufs: Vec<u64> = (0..4).map(|_| g.alloc(128, 8)).collect();
        let heads: Vec<u16> = bufs.iter().map(|&b| rx.submit(&g, &[Buf::inp(b, 128)])).collect();
        let payload: Vec<u8> = (0..286u32).map(|i| i as u8).collect();
        let f = frame(OUR_MAC, &payload);
        assert_eq!(f.len(), 300);
        assert_eq!(receive(&g, 0, &f), RxOutcome::Delivered);

        let used: Vec<(u32, u32)> = std::iter::from_fn(|| rx.get_used(&g)).collect();
        let want = [(heads[0], 128), (heads[1], 128), (heads[2], 312 - 256)];
        assert_eq!(used, want.map(|(h, l)| (u32::from(h), l)));
        assert_eq!(g.read_u16(bufs[0] + 10), 3);
        let mut got = g.read_mem(bufs[0] + 12, 116);
        got.extend(g.read_mem(bufs[1], 128));
        got.extend(g.read_mem(bufs[2], 56));
        assert_eq!(got, f);

        // One buffer left is not enough for another one.
        assert_eq!(receive(&g, 0, &f), RxOutcome::NoBuffers);
    }
}

#[test]
fn non_mergeable_rx_needs_one_big_enough_chain() {
    let peer = Loopback::default();
    let wanted = features() & !feature(VIRTIO_NET_F_MRG_RXBUF);
    let (mut g, mut rx, _tx) = start(false, VirtioNetConf::default(), &peer, wanted);
    assert_eq!(dev(&g, |_, d| d.guest_hdr_len()), 12);
    let f = frame(OUR_MAC, &[7; 86]);

    // Too small: the frame is dropped and the buffer stays with the device.
    let small = g.alloc(64, 8);
    rx.submit(&g, &[Buf::inp(small, 64)]);
    assert_eq!(receive(&g, 0, &f), RxOutcome::Dropped);
    assert_eq!(rx.used_idx(&g), 0);

    // A header descriptor and a data descriptor in one chain.
    let peer = Loopback::default();
    let (mut g, mut rx, _tx) = start(false, VirtioNetConf::default(), &peer, wanted);
    let hdr = g.alloc(12, 8);
    let data = g.alloc(200, 8);
    let head = rx.submit(&g, &[Buf::inp(hdr, 12), Buf::inp(data, 200)]);
    assert_eq!(receive(&g, 0, &f), RxOutcome::Delivered);
    assert_eq!(rx.get_used(&g), Some((u32::from(head), 12 + 100)));
    assert_eq!(g.read_mem(data, 100), f);
}

/// Frames that find no buffers wait, and the peer hears when the driver adds some. This is
/// what `rx_stop_cont_test()` checks with a stopped VM.
#[test]
fn rx_waits_for_buffers() {
    let peer = Loopback::default();
    let (mut g, mut rx, _tx) = start(false, VirtioNetConf::default(), &peer, features());
    peer.ready.lock().unwrap().clear();
    assert_eq!(receive(&g, 0, b"TEST\0"), RxOutcome::NoBuffers);
    let buf = g.alloc(64, 8);
    rx.submit(&g, &[Buf::inp(buf, 64)]);
    g.kick(&rx);
    assert_eq!(peer.ready.lock().unwrap().as_slice(), &[0]);
    assert_eq!(receive(&g, 0, b"TEST\0"), RxOutcome::Delivered);
    assert_eq!(g.read_mem(buf + 12, 5), b"TEST\0");

    // Before DRIVER_OK nothing is received.
    let mut g = Guest::new(false, net(VirtioNetConf::default(), &peer));
    g.negotiate(features());
    let _rx = g.setup_queue(0, 0);
    assert_eq!(receive(&g, 0, b"TEST"), RxOutcome::NotReady);
}

#[test]
fn rx_filter_modes() {
    let peer = Loopback::default();
    let (mut g, mut rx, _tx) = start(false, VirtioNetConf::default(), &peer, features());
    let mut cq = g.setup_queue(2, 0);
    let filter = |g: &Guest, f: &[u8]| dev(g, |_, d| d.receive_filter(f));
    let ours = frame(OUR_MAC, b"x");
    let other = frame(OTHER_MAC, b"x");
    let bcast = frame(BCAST, b"x");
    let mcast = frame(MCAST, b"x");

    // Promiscuous by default.
    assert!(dev(&g, |_, d| d.promisc()));
    assert!(filter(&g, &other) && filter(&g, &mcast));

    assert_eq!(ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_RX, VIRTIO_NET_CTRL_RX_PROMISC, &[0]), 0);
    assert!(filter(&g, &ours) && filter(&g, &bcast));
    assert!(!filter(&g, &other) && !filter(&g, &mcast));
    assert!(!filter(&g, &ours[..13]), "a runt frame never passes");

    // A filtered frame is consumed without touching the buffers.
    let buf = g.alloc(64, 8);
    rx.submit(&g, &[Buf::inp(buf, 64)]);
    assert_eq!(receive(&g, 0, &other), RxOutcome::Filtered);
    assert_eq!(receive(&g, 0, &ours), RxOutcome::Delivered);

    let set = |g: &mut Guest, cq: &mut common::SplitRing, cmd: u8, on: u8| {
        assert_eq!(ctrl(g, cq, VIRTIO_NET_CTRL_RX, cmd, &[on]), VIRTIO_NET_OK);
    };
    set(&mut g, &mut cq, VIRTIO_NET_CTRL_RX_ALLMULTI, 1);
    assert!(filter(&g, &mcast));
    set(&mut g, &mut cq, VIRTIO_NET_CTRL_RX_NOMULTI, 1);
    assert!(!filter(&g, &mcast), "nomulti wins over allmulti");
    set(&mut g, &mut cq, VIRTIO_NET_CTRL_RX_NOMULTI, 0);
    set(&mut g, &mut cq, VIRTIO_NET_CTRL_RX_ALLMULTI, 0);
    set(&mut g, &mut cq, VIRTIO_NET_CTRL_RX_NOBCAST, 1);
    assert!(!filter(&g, &bcast));
    set(&mut g, &mut cq, VIRTIO_NET_CTRL_RX_NOBCAST, 0);
    set(&mut g, &mut cq, VIRTIO_NET_CTRL_RX_ALLUNI, 1);
    assert!(filter(&g, &other));
    set(&mut g, &mut cq, VIRTIO_NET_CTRL_RX_NOUNI, 1);
    assert!(!filter(&g, &other) && !filter(&g, &ours));
    set(&mut g, &mut cq, VIRTIO_NET_CTRL_RX_NOUNI, 0);
    set(&mut g, &mut cq, VIRTIO_NET_CTRL_RX_ALLUNI, 0);

    // The MAC table.
    let table = mac_table(&[OTHER_MAC], &[MCAST]);
    assert_eq!(
        ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_MAC, VIRTIO_NET_CTRL_MAC_TABLE_SET, &table),
        0
    );
    assert_eq!(
        dev(&g, |_, d| d.mac_table().clone()),
        MacTable {
            macs: vec![OTHER_MAC, MCAST],
            first_multi: 1,
            uni_overflow: false,
            multi_overflow: false
        }
    );
    assert!(filter(&g, &other) && filter(&g, &mcast));
    assert!(!filter(&g, &frame([0x01, 0, 0x5e, 0, 0, 2], b"x")));
    assert!(!filter(&g, &frame([0x52, 0, 0, 0, 0, 1], b"x")));

    // Too many unicast addresses: all unicast frames pass.
    let many: Vec<[u8; 6]> = (0..65u8).map(|i| [0x52, 0, 0, 0, 1, i]).collect();
    let table = mac_table(&many, &[]);
    assert_eq!(
        ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_MAC, VIRTIO_NET_CTRL_MAC_TABLE_SET, &table),
        0
    );
    let t = dev(&g, |_, d| d.mac_table().clone());
    assert!(t.uni_overflow && !t.multi_overflow && t.macs.is_empty());
    assert!(filter(&g, &frame([0x52, 9, 9, 9, 9, 9], b"x")));
    assert!(!filter(&g, &mcast));

    // Multicast overflow counts what the unicast part already took.
    let table = mac_table(&many[..60], &many[..5]);
    assert_eq!(
        ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_MAC, VIRTIO_NET_CTRL_MAC_TABLE_SET, &table),
        0
    );
    let t = dev(&g, |_, d| d.mac_table().clone());
    assert!(!t.uni_overflow && t.multi_overflow && t.macs.len() == 60);
    assert!(filter(&g, &mcast));
}

#[test]
fn vlan_filter() {
    let peer = Loopback::default();
    let (mut g, _rx, _tx) = start(false, VirtioNetConf::default(), &peer, features());
    let mut cq = g.setup_queue(2, 0);
    let mut tagged = OUR_MAC.to_vec();
    tagged.extend_from_slice(&OTHER_MAC);
    tagged.extend_from_slice(&[0x81, 0x00, 0x20, 0x05, 0x08, 0x00]);
    let filter = |g: &Guest, f: &[u8]| dev(g, |_, d| d.receive_filter(f));

    assert_eq!(ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_RX, VIRTIO_NET_CTRL_RX_PROMISC, &[0]), 0);
    // CTRL_VLAN was negotiated, so every VLAN starts out filtered.
    assert!(!dev(&g, |_, d| d.vlan_allowed(5)));
    assert!(!filter(&g, &tagged));
    assert!(!filter(&g, &tagged[..15]));
    let add = VIRTIO_NET_CTRL_VLAN_ADD;
    let del = VIRTIO_NET_CTRL_VLAN_DEL;
    assert_eq!(ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_VLAN, add, &5u16.to_le_bytes()), 0);
    assert!(filter(&g, &tagged));
    assert_eq!(ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_VLAN, del, &5u16.to_le_bytes()), 0);
    assert!(!filter(&g, &tagged));
    assert_eq!(ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_VLAN, add, &4096u16.to_le_bytes()), 1);
    assert_eq!(ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_VLAN, 2, &5u16.to_le_bytes()), 1);

    // Without CTRL_VLAN every VLAN passes.
    let (g, _rx, _tx) = start(
        false,
        VirtioNetConf::default(),
        &peer,
        features() & !feature(VIRTIO_NET_F_CTRL_VLAN),
    );
    assert!(dev(&g, |_, d| d.vlan_allowed(5) && d.vlan_allowed(4095)));
}

#[test]
fn ctrl_commands_and_errors() {
    for legacy in [true, false] {
        let peer = Loopback::default();
        let (mut g, _rx, _tx) = start(legacy, VirtioNetConf::default(), &peer, features());
        let mut cq = g.setup_queue(2, 0);
        let ok = VIRTIO_NET_OK;
        let err = VIRTIO_NET_ERR;

        assert_eq!(ctrl(&mut g, &mut cq, 42, 0, &[0]), err);
        assert_eq!(ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_RX, 6, &[1]), err);
        assert_eq!(ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_RX, VIRTIO_NET_CTRL_RX_PROMISC, &[]), err);

        // MAC address set.
        let set = VIRTIO_NET_CTRL_MAC_ADDR_SET;
        assert_eq!(ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_MAC, set, &OTHER_MAC[..5]), err);
        assert_eq!(ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_MAC, set, &[0; 7]), err);
        assert_eq!(ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_MAC, set, &OTHER_MAC), ok);
        assert_eq!(dev(&g, |_, d| d.mac()), OTHER_MAC);
        assert_eq!(g.config_readb(5), OTHER_MAC[5]);
        assert_eq!(ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_MAC, 2, &OTHER_MAC), err);

        // MAC table sizes that do not add up.
        let table_set = VIRTIO_NET_CTRL_MAC_TABLE_SET;
        let mut bad = mac_table(&[OTHER_MAC], &[MCAST]);
        bad.push(0);
        assert_eq!(ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_MAC, table_set, &bad), err);
        let short = &mac_table(&[OTHER_MAC], &[MCAST])[..19];
        assert_eq!(ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_MAC, table_set, short), err);
        assert_eq!(ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_MAC, table_set, &[9, 0, 0, 0]), err);
        assert_eq!(ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_MAC, table_set, &[0, 0, 0, 0]), err);

        // Announce ack without an announce.
        let ack = VIRTIO_NET_CTRL_ANNOUNCE_ACK;
        assert_eq!(ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_ANNOUNCE, ack, &[]), err);

        // No MQ, RSS or offloads without a vnet header peer.
        let pairs = VIRTIO_NET_CTRL_MQ_VQ_PAIRS_SET;
        assert_eq!(ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_MQ, pairs, &1u16.to_le_bytes()), err);
        let rss = VIRTIO_NET_CTRL_MQ_RSS_CONFIG;
        assert_eq!(ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_MQ, rss, &[0; 16]), err);
        let offloads = VIRTIO_NET_CTRL_GUEST_OFFLOADS_SET;
        assert_eq!(ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_GUEST_OFFLOADS, offloads, &[0; 8]), err);
        assert!(!dev(&g, |vdev, _| vdev.is_broken()));

        // A command without room for the ack breaks the device.
        let o = g.alloc(2, 8);
        g.write_mem(o, &[VIRTIO_NET_CTRL_RX, 0]);
        cq.submit(&g, &[Buf::out(o, 2)]);
        g.kick(&cq);
        assert!(dev(&g, |vdev, _| vdev.is_broken()));
        if !legacy {
            assert_ne!(g.status() & VIRTIO_CONFIG_S_NEEDS_RESET, 0);
        }
    }
}

#[test]
fn announce_and_ack() {
    let peer = Loopback::default();
    let (mut g, _rx, _tx) = start(false, VirtioNetConf::default(), &peer, features());
    let mut cq = g.setup_queue(2, 0);
    g.ack();
    assert!(dev(&g, |vdev, d| d.announce(vdev)));
    assert_eq!(g.config_readw(6), VIRTIO_NET_S_LINK_UP | VIRTIO_NET_S_ANNOUNCE);
    assert_ne!(g.isr() & 2, 0, "config interrupt");
    let ack = VIRTIO_NET_CTRL_ANNOUNCE_ACK;
    assert_eq!(ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_ANNOUNCE, ack, &[]), VIRTIO_NET_OK);
    assert_eq!(g.config_readw(6), VIRTIO_NET_S_LINK_UP);

    // Not without GUEST_ANNOUNCE.
    let wanted = features() & !feature(VIRTIO_NET_F_GUEST_ANNOUNCE);
    let (g, _rx, _tx) = start(false, VirtioNetConf::default(), &peer, wanted);
    assert!(!dev(&g, |vdev, d| d.announce(vdev)));
    assert_eq!(g.config_readw(6), VIRTIO_NET_S_LINK_UP);
}

#[test]
fn link_status() {
    let peer = Loopback::default();
    let (mut g, mut rx, mut tx) = start(false, VirtioNetConf::default(), &peer, features());
    g.ack();
    dev(&g, |vdev, d| d.set_link_status(vdev, false));
    assert_eq!(g.config_readw(6), 0);
    assert_ne!(g.isr() & 2, 0);
    g.ack();
    // No change, no interrupt.
    dev(&g, |vdev, d| d.set_link_status(vdev, false));
    assert_eq!(g.isr(), 0);

    let buf = g.alloc(64, 8);
    rx.submit(&g, &[Buf::inp(buf, 64)]);
    assert_eq!(receive(&g, 0, b"TEST"), RxOutcome::Dropped);
    assert_eq!(rx.used_idx(&g), 0);

    // What the guest sends while the link is down is thrown away.
    let req = g.alloc(64, 8);
    let head = tx.submit(&g, &[Buf::out(req, 64)]);
    g.kick(&tx);
    assert_eq!(tx.get_used(&g), Some((u32::from(head), 0)));
    assert!(peer.frames().is_empty());

    dev(&g, |vdev, d| d.set_link_status(vdev, true));
    assert_eq!(g.config_readw(6), VIRTIO_NET_S_LINK_UP);
    assert_eq!(receive(&g, 0, b"TEST"), RxOutcome::Delivered);
}

#[test]
fn tx_waits_for_a_busy_peer() {
    let peer = Loopback::default();
    let (mut g, _rx, mut tx) = start(false, VirtioNetConf::default(), &peer, features());
    peer.busy.store(true, Ordering::SeqCst);
    let req = g.alloc(64, 8);
    g.write_mem(req + 12, b"ONE");
    tx.submit(&g, &[Buf::out(req, 15)]);
    g.kick(&tx);
    assert_eq!(tx.used_idx(&g), 0);
    assert!(peer.frames().is_empty());

    peer.busy.store(false, Ordering::SeqCst);
    // A kick while waiting does nothing, the resume does the work.
    g.kick(&tx);
    assert_eq!(tx.used_idx(&g), 0);
    dev(&g, |vdev, d| d.tx_resume(vdev, 0));
    assert_eq!(tx.used_idx(&g), 1);
    assert_eq!(peer.frames(), vec![b"ONE".to_vec()]);
}

#[test]
fn vnet_header_peer() {
    let peer = Loopback { vnet_hdr: true, ..Loopback::default() };
    let (mut g, mut rx, mut tx) = start(false, VirtioNetConf::default(), &peer, features());
    let mut cq = g.setup_queue(2, 0);
    assert_eq!(dev(&g, |_, d| d.host_hdr_len()), 12);
    let negotiated = dev(&g, |_, d| d.curr_guest_offloads());
    assert_ne!(negotiated & feature(VIRTIO_NET_F_GUEST_CSUM), 0);
    assert_eq!(peer.offloads.lock().unwrap().last(), Some(&negotiated));

    // The guest's header goes to the peer.
    let hdr = VirtioNetHdr {
        flags: VIRTIO_NET_HDR_F_NEEDS_CSUM,
        csum_start: 34,
        csum_offset: 16,
        ..VirtioNetHdr::default()
    };
    let req = g.alloc(64, 8);
    g.write_mem(req, &hdr.to_bytes());
    g.write_mem(req + 12, b"DATA");
    tx.submit(&g, &[Buf::out(req, 16)]);
    g.kick(&tx);
    let sent = peer.sent.lock().unwrap().clone();
    assert_eq!(sent, vec![(0, Some(hdr), b"DATA".to_vec())]);

    // And the peer's header goes to the guest, with num_buffers filled in.
    let buf = g.alloc(64, 8);
    rx.submit(&g, &[Buf::inp(buf, 64)]);
    let rx_hdr = VirtioNetHdr { flags: VIRTIO_NET_HDR_F_DATA_VALID, ..VirtioNetHdr::default() };
    let out = dev(&g, |vdev, d| d.receive_on(vdev, 0, Some(&rx_hdr), b"PKT"));
    assert_eq!(out, RxOutcome::Delivered);
    let got = VirtioNetHdr::from_bytes(&g.read_mem(buf, 12)).unwrap();
    assert_eq!(got, VirtioNetHdr { num_buffers: 1, ..rx_hdr });

    // Offloads through the control queue: only a subset of what was negotiated.
    let set = VIRTIO_NET_CTRL_GUEST_OFFLOADS_SET;
    let csum = feature(VIRTIO_NET_F_GUEST_CSUM).to_le_bytes();
    assert_eq!(ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_GUEST_OFFLOADS, set, &csum), 0);
    assert_eq!(dev(&g, |_, d| d.curr_guest_offloads()), feature(VIRTIO_NET_F_GUEST_CSUM));
    assert_eq!(peer.offloads.lock().unwrap().last(), Some(&feature(VIRTIO_NET_F_GUEST_CSUM)));
    let ufo = feature(VIRTIO_NET_F_GUEST_UFO).to_le_bytes();
    assert_eq!(ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_GUEST_OFFLOADS, set, &ufo), 1);
    let rsc = feature(VIRTIO_NET_F_RSC_EXT).to_le_bytes();
    assert_eq!(ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_GUEST_OFFLOADS, set, &rsc), 0);
    assert_eq!(dev(&g, |_, d| d.curr_guest_offloads()), 0);
}

#[test]
fn multiqueue() {
    let peer = Loopback::default();
    let conf = VirtioNetConf {
        queue_pairs: 4,
        host_features: VIRTIO_NET_DEFAULT_HOST_FEATURES | feature(VIRTIO_NET_F_MQ),
        ..VirtioNetConf::default()
    };
    let mut g = Guest::new(false, net(conf.clone(), &peer));
    assert_eq!(dev(&g, |vdev, _| vdev.num_queues()), 3);
    assert_eq!(g.config_readw(8), 4);
    assert_eq!(dev(&g, |_, d| d.max_queue_pairs()), 4);

    let accepted = g.negotiate(features());
    assert!(has_feature(accepted, VIRTIO_NET_F_MQ));
    // rx0, tx0, ..., rx3, tx3, ctrl.
    assert_eq!(dev(&g, |vdev, _| vdev.num_queues()), 9);
    assert_eq!(dev(&g, |vdev, _| vdev.queue_num_max(8)), VIRTIO_NET_CTRL_QUEUE_SIZE);
    assert_eq!(dev(&g, |vdev, _| vdev.queue_num_max(7)), 256);
    let mut rings: Vec<common::SplitRing> = (0..8).map(|i| g.setup_queue(i, 0)).collect();
    let mut cq = g.setup_queue(8, 0);
    g.driver_ok();
    assert_eq!(dev(&g, |_, d| (d.multiqueue(), d.curr_queue_pairs())), (true, 1));

    let pairs = VIRTIO_NET_CTRL_MQ_VQ_PAIRS_SET;
    for bad in [0u16, 5, 0x8001] {
        assert_eq!(ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_MQ, pairs, &bad.to_le_bytes()), 1);
    }
    assert_eq!(ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_MQ, pairs, &[2]), 1);
    peer.ready.lock().unwrap().clear();
    assert_eq!(ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_MQ, pairs, &2u16.to_le_bytes()), 0);
    assert_eq!(dev(&g, |_, d| d.curr_queue_pairs()), 2);
    assert_eq!(peer.ready.lock().unwrap().as_slice(), &[0, 1]);

    // Pair 1 receives and sends, pair 3 is not in use.
    let buf = g.alloc(64, 8);
    rings[2].submit(&g, &[Buf::inp(buf, 64)]);
    assert_eq!(receive(&g, 1, b"PAIR1"), RxOutcome::Delivered);
    assert_eq!(g.read_mem(buf + 12, 5), b"PAIR1");
    assert_eq!(receive(&g, 3, b"PAIR3"), RxOutcome::NotReady);
    let req = g.alloc(64, 8);
    g.write_mem(req + 12, b"OUT");
    rings[3].submit(&g, &[Buf::out(req, 15)]);
    g.kick(&rings[3]);
    assert_eq!(peer.sent.lock().unwrap()[0].0, 1);

    // A reset goes back to one pair in use, and renegotiating without MQ drops the others.
    g.set_status(0);
    assert_eq!(dev(&g, |_, d| d.curr_queue_pairs()), 1);
    let mut g2 = Guest::new(false, net(conf, &peer));
    g2.negotiate(features() & !feature(VIRTIO_NET_F_MQ));
    assert_eq!(dev(&g2, |vdev, d| (vdev.num_queues(), d.multiqueue())), (3, false));
    let mut cq2 = g2.setup_queue(2, 0);
    g2.driver_ok();
    assert_eq!(ctrl(&mut g2, &mut cq2, VIRTIO_NET_CTRL_MQ, pairs, &1u16.to_le_bytes()), 1);
}

#[test]
fn legacy_config_write_sets_the_mac() {
    let peer = Loopback::default();
    let g = Guest::new(true, net(VirtioNetConf::default(), &peer));
    g.negotiate(features() & !feature(VIRTIO_NET_F_CTRL_MAC_ADDR));
    for (i, b) in OTHER_MAC.iter().enumerate() {
        g.store(VIRTIO_MMIO_CONFIG + i as u64, 1, u64::from(*b));
    }
    assert_eq!(dev(&g, |_, d| d.mac()), OTHER_MAC);

    // Not when the MAC is set through the control queue.
    let g = Guest::new(true, net(VirtioNetConf::default(), &peer));
    g.negotiate(features());
    g.store(VIRTIO_MMIO_CONFIG, 1, 0x02);
    assert_eq!(dev(&g, |_, d| d.mac()), OUR_MAC);
}

#[test]
fn reset_restores_the_filter() {
    let peer = Loopback::default();
    let (mut g, _rx, _tx) = start(false, VirtioNetConf::default(), &peer, features());
    let mut cq = g.setup_queue(2, 0);
    ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_RX, VIRTIO_NET_CTRL_RX_PROMISC, &[0]);
    ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_RX, VIRTIO_NET_CTRL_RX_NOBCAST, &[1]);
    ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_MAC, VIRTIO_NET_CTRL_MAC_ADDR_SET, &OTHER_MAC);
    let table = mac_table(&[OTHER_MAC], &[]);
    ctrl(&mut g, &mut cq, VIRTIO_NET_CTRL_MAC, VIRTIO_NET_CTRL_MAC_TABLE_SET, &table);
    g.set_status(0);
    dev(&g, |_, d| {
        assert!(d.promisc() && !d.nobcast());
        assert_eq!(d.mac(), OUR_MAC);
        assert_eq!(d.mac_table(), &MacTable::default());
    });
}

#[test]
fn header_round_trip() {
    let h = VirtioNetHdr {
        flags: 1,
        gso_type: VIRTIO_NET_HDR_GSO_TCPV4,
        hdr_len: 54,
        gso_size: 1448,
        csum_start: 34,
        csum_offset: 16,
        num_buffers: 2,
    };
    assert_eq!(VirtioNetHdr::from_bytes(&h.to_bytes()), Some(h));
    assert_eq!(
        VirtioNetHdr::from_bytes(&h.to_bytes()[..10]),
        Some(VirtioNetHdr { num_buffers: 0, ..h })
    );
    assert_eq!(VirtioNetHdr::from_bytes(&[0; 9]), None);
}
