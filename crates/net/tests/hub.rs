// SPDX-License-Identifier: GPL-2.0-or-later

//! Hubs, `-netdev hubport`, `-net` and `-nic`, and `info network`.

mod common;

use std::sync::Arc;
use std::sync::atomic::Ordering;

use common::{Recorder, frame, netdev};
use ruvm_net::{MacAddr, Net, NetClient, Nic, NicConf, hub_id_for_client};

fn nic_on(net: &mut Net, peer: &Arc<NetClient>, name: &str) -> (Arc<Nic>, Arc<Recorder>) {
    let rec = Recorder::new();
    let r = rec.clone();
    let conf =
        NicConf { macaddr: MacAddr([0x52, 0x54, 0, 0x12, 0x34, 0x56]), peers: vec![peer.clone()] };
    let nic = net.new_nic(&conf, "e1000", Some(name), move |_, _| r.clone());
    (nic, rec)
}

#[test]
fn hub_forwards_to_every_other_port() {
    let mut net = Net::new();
    netdev(&mut net, "hubport,id=p0,hubid=3").unwrap();
    netdev(&mut net, "hubport,id=p1,hubid=3").unwrap();
    netdev(&mut net, "hubport,id=p2,hubid=3").unwrap();
    let p: Vec<_> = ["p0", "p1", "p2"].iter().map(|n| net.find_netdev(n).unwrap()).collect();
    for port in &p {
        assert_eq!(hub_id_for_client(port), Some(3));
    }
    let (n0, r0) = nic_on(&mut net, &p[0], "n0");
    let (_n1, r1) = nic_on(&mut net, &p[1], "n1");
    let (_n2, r2) = nic_on(&mut net, &p[2], "n2");
    assert_eq!(hub_id_for_client(n0.queue()), Some(3));

    assert_eq!(n0.queue().send_packet(&frame(1, 64)), 64);
    assert_eq!(r0.count(), 0);
    assert_eq!(r1.take(), vec![frame(1, 64)]);
    assert_eq!(r2.take(), vec![frame(1, 64)]);

    assert_eq!(net.hubs().len(), 1);
    assert_eq!(net.hubs()[0].ports().len(), 3);
    assert!(
        net.check_clients().is_empty()
            || net.check_clients().iter().all(|w| !w.contains("no peer"))
    );
}

#[test]
fn hub_waits_for_all_ports() {
    let mut net = Net::new();
    netdev(&mut net, "hubport,id=p0,hubid=0").unwrap();
    netdev(&mut net, "hubport,id=p1,hubid=0").unwrap();
    let p0 = net.find_netdev("p0").unwrap();
    let p1 = net.find_netdev("p1").unwrap();
    let (n0, _r0) = nic_on(&mut net, &p0, "n0");
    let (n1, r1) = nic_on(&mut net, &p1, "n1");
    r1.closed.store(true, Ordering::SeqCst);
    assert!(!n0.queue().can_send_packet());
    n0.queue().send_packet(&frame(2, 64));
    assert_eq!(r1.count(), 0);
    r1.closed.store(false, Ordering::SeqCst);
    n1.queue().flush_queued_packets();
    assert_eq!(r1.take(), vec![frame(2, 64)]);
}

#[test]
fn hubport_with_netdev_peer() {
    let mut net = Net::new();
    netdev(&mut net, "hubport,id=a,hubid=1").unwrap();
    netdev(&mut net, "hubport,id=b,hubid=2,netdev=a").unwrap();
    let a = net.find_netdev("a").unwrap();
    let b = net.find_netdev("b").unwrap();
    assert!(Arc::ptr_eq(&a.peer().unwrap(), &b));
    assert_eq!(
        common::netdev_err("hubport,id=c,hubid=1,netdev=missing"),
        "netdev 'missing' not found"
    );
}

#[test]
fn info_network_lists_hubs_and_pairs() {
    let mut net = Net::new();
    netdev(&mut net, "hubport,id=hp,hubid=5").unwrap();
    let hp = net.find_netdev("hp").unwrap();
    let (n, _r) = nic_on(&mut net, &hp, "nic0");
    n.queue().set_info_str("model=e1000,macaddr=52:54:00:12:34:56");
    let (lone, _) = common::plain_client(&mut net, "lone");
    lone.set_info_str("x");
    let out = net.info_network();
    assert_eq!(
        out,
        "hub 5\n \\ hp: nic0: index=0,type=nic,model=e1000,macaddr=52:54:00:12:34:56\nlone: index=0,type=socket,x\n"
    );
}

#[test]
fn legacy_net_options_go_through_hub_zero() {
    let mut net = Net::new();
    net.parse_net("nic,model=e1000,macaddr=52:54:00:12:34:99").unwrap();
    net.init_clients().unwrap();
    assert_eq!(net.nb_nics(), 1);
    let nd = &net.nd_table()[0];
    assert!(nd.used);
    assert_eq!(nd.model.as_deref(), Some("e1000"));
    assert_eq!(nd.macaddr, MacAddr([0x52, 0x54, 0, 0x12, 0x34, 0x99]));
    let port = nd.netdev.clone().unwrap();
    assert_eq!(hub_id_for_client(&port), Some(0));
    assert_eq!(port.name(), "hub0port0");
}

#[test]
fn hubport_is_rejected_with_net() {
    let mut net = Net::new();
    net.parse_net("hubport,hubid=1,id=x").unwrap();
    let e = net.init_clients().unwrap_err();
    assert_eq!(e.message(), "network backend 'hubport' is only supported with -netdev/-nic");
}

#[test]
fn user_backend_is_not_compiled() {
    let mut net = Net::new();
    net.parse_net("user,hostfwd=tcp::2222-:22").unwrap();
    let e = net.init_clients().unwrap_err();
    assert_eq!(e.message(), "network backend 'user' is not compiled into this binary");

    assert_eq!(
        common::netdev_err("user,id=u0,net=10.0.2.0/24"),
        "network backend 'user' is not compiled into this binary"
    );
}

#[test]
fn nic_option_records_a_nic() {
    let mut net = Net::new();
    net.parse_nic("hubport,hubid=4,model=virtio-net-pci,mac=52:54:00:12:34:77").unwrap();
    net.init_clients().unwrap();
    assert_eq!(net.nb_nics(), 1);
    let nd = net.nd_table()[0].clone();
    assert_eq!(nd.model.as_deref(), Some("virtio-net-pci"));
    let backend = nd.netdev.unwrap();
    assert_eq!(hub_id_for_client(&backend), Some(4));
    assert!(net.find_nic_info("virtio-net-pci", false, None).is_some());
    assert!(net.find_nic_info("e1000", false, None).is_none());

    let mut net = Net::new();
    net.parse_nic("none").unwrap();
    net.init_clients().unwrap();
    assert_eq!(net.nb_nics(), 0);

    let mut net = Net::new();
    net.parse_nic("hubport,hubid=0,mac=01:00:00:00:00:00").unwrap();
    assert_eq!(net.init_clients().unwrap_err().message(), "NIC cannot have multicast MAC address");
}

#[test]
fn check_clients_warns() {
    let mut net = Net::new();
    netdev(&mut net, "hubport,id=p,hubid=7").unwrap();
    let w = net.check_clients();
    assert!(w.contains(&"hub port p has no peer".to_string()), "{w:?}");
    assert!(w.contains(&"netdev p has no peer".to_string()), "{w:?}");

    let mut net = Net::new();
    net.parse_net("nic,model=e1000").unwrap();
    net.init_clients().unwrap();
    let w = net.check_clients();
    // The NIC was never created, so the hub port still waits for its peer.
    assert!(w.contains(&"hub port hub0port0 has no peer".to_string()), "{w:?}");
    assert!(
        w.iter().any(|s| s.starts_with("requested NIC (")
            && s.ends_with("model e1000) was not created (not supported by this machine?)")),
        "{w:?}"
    );
}

#[test]
fn netdev_del_and_duplicates() {
    let mut net = Net::new();
    netdev(&mut net, "hubport,id=p,hubid=0").unwrap();
    assert_eq!(
        net.netdev_add_opts("hubport,id=p,hubid=0").unwrap_err().message(),
        "Duplicate ID 'p' for netdev"
    );
    net.netdev_add_opts("hubport,id=q,hubid=0").unwrap();
    net.netdev_del("q").unwrap();
    assert!(net.find_netdev("q").is_none());
    assert_eq!(net.netdev_del("q").unwrap_err().message(), "Device 'q' not found");
    // A port made for -net is not a netdev.
    net.hub_add_port(9, Some("notnetdev"), None);
    assert_eq!(
        net.netdev_del("notnetdev").unwrap_err().message(),
        "Device 'notnetdev' is not a netdev"
    );
}
