// SPDX-License-Identifier: GPL-2.0-or-later

//! Option parsing and the errors QEMU gives for bad `-netdev` options.

mod common;

use common::netdev_err;
use ruvm_net::{Net, netdev_is_modern, parse_modern};

#[test]
fn generic_errors() {
    assert_eq!(netdev_err("hubport,hubid=0"), "Parameter 'id' is missing");
    assert_eq!(netdev_err("hubport,id=h,hubid=0,bogus=1"), "Invalid parameter 'bogus'");
    assert_eq!(netdev_err("nosuch,id=x"), "Parameter 'type' does not accept value 'nosuch'");
    assert_eq!(netdev_err("nic,id=x"), "network backend 'nic' is not compiled into this binary");
    assert_eq!(netdev_err("vde,id=x"), "network backend 'vde' is not compiled into this binary");
    assert_eq!(netdev_err("user,id=x"), "network backend 'user' is not compiled into this binary");

    let mut net = Net::new();
    net.parse_netdev("hubport,id=d,hubid=0").unwrap();
    let e = net.parse_netdev("hubport,id=d,hubid=1").unwrap_err();
    assert_eq!(e.message(), "Duplicate ID 'd' for netdev");
}

#[test]
fn duplicate_id_across_kinds() {
    let mut net = Net::new();
    net.parse_netdev("hubport,id=d,hubid=0").unwrap();
    net.parse_netdev("stream,id=d,addr.type=unix,addr.path=/nonexistent/x").unwrap();
    assert_eq!(net.init_clients().unwrap_err().message(), "Duplicate ID 'd'");
}

#[test]
fn modern_detection() {
    assert!(netdev_is_modern("stream,id=s,addr.type=inet,addr.host=localhost,addr.port=1"));
    assert!(netdev_is_modern("dgram,id=d"));
    assert!(netdev_is_modern("{\"type\":\"tap\",\"id\":\"t\"}"));
    assert!(!netdev_is_modern("tap,id=t"));
    assert!(!netdev_is_modern("socket,id=s,listen=:1"));
    let nd =
        parse_modern("stream,id=s0,server=on,addr.type=inet,addr.host=127.0.0.1,addr.port=5555")
            .unwrap();
    assert_eq!(nd.id, "s0");
}

#[test]
fn tap_errors() {
    assert_eq!(
        netdev_err("tap,id=t,vhost=off,vhostfd=3"),
        "vhostfd(s)= is not valid without vhost"
    );
    assert_eq!(
        netdev_err("tap,id=t,fd=3,queues=2"),
        "queues=, helper=, fds= and fd= are mutual exclusive"
    );
    assert_eq!(
        netdev_err("tap,id=t,fd=3,ifname=tap0"),
        "ifname=, script=, downscript=, vnet_hdr= are invalid with fd=/fds=/helper="
    );
    assert_eq!(
        netdev_err("tap,id=t,fds=3,vnet_hdr=on"),
        "ifname=, script=, downscript=, vnet_hdr= are invalid with fd=/fds=/helper="
    );
    assert_eq!(netdev_err("tap,id=t,queues=0"), "queues must be greater than zero");
    assert_eq!(netdev_err("tap,id=t,fd=abc"), "Invalid file descriptor number 'abc'");
    assert_eq!(netdev_err("tap,id=t,fd=3:4"), "expected 1 socket fds, got 2");
    assert_eq!(
        netdev_err("tap,id=t,helper=/bin/true"),
        "the bridge helper (helper=) is not supported"
    );

    let mut net = Net::new();
    net.parse_net("tap,queues=2").unwrap();
    assert_eq!(
        net.init_clients().unwrap_err().message(),
        "Multiqueue tap cannot be used with hubs"
    );
}

#[cfg(not(target_os = "linux"))]
#[test]
fn tap_creation_needs_linux() {
    assert_eq!(
        netdev_err("tap,id=t,script=no,downscript=no"),
        "tap interfaces can only be created on Linux hosts; use fd= instead"
    );
}

#[test]
fn socket_errors() {
    let one = "exactly one of listen=, connect=, mcast= or udp= is required";
    assert_eq!(netdev_err("socket,id=s"), one);
    assert_eq!(netdev_err("socket,id=s,listen=:1,connect=:2"), one);
    assert_eq!(
        netdev_err("socket,id=s,listen=:1,localaddr=127.0.0.1:0"),
        "localaddr= is only valid with mcast= or udp="
    );
    assert_eq!(netdev_err("socket,id=s,udp=127.0.0.1:1"), "localaddr= is mandatory with udp=");
    assert_eq!(
        netdev_err("socket,id=s,mcast=230.0.0.1:1234,localaddr=nothing"),
        "localaddr 'nothing' is not a valid IPv4 address"
    );
    assert_eq!(
        netdev_err("socket,id=s,connect=127.0.0.1"),
        "host address '127.0.0.1' doesn't contain ':' separating host from port"
    );
    assert_eq!(netdev_err("socket,id=s,fd=x1"), "Invalid file descriptor number 'x1'");
}

#[test]
fn stream_errors() {
    assert_eq!(
        netdev_err(
            "stream,id=s,server=on,reconnect-ms=10,addr.type=inet,addr.host=127.0.0.1,addr.port=1"
        ),
        "'reconnect-ms' option is incompatible with socket in server mode"
    );
    assert_eq!(netdev_err("stream,id=s,server=on"), "Parameter 'addr' is missing");
}

#[test]
fn dgram_errors() {
    assert_eq!(
        netdev_err("dgram,id=d,remote.type=inet,remote.host=127.0.0.1,remote.port=1"),
        "dgram requires local= parameter"
    );
    assert_eq!(
        netdev_err("dgram,id=d,local.type=inet,local.host=127.0.0.1,local.port=1"),
        "type=inet or type=unix requires remote parameter"
    );
    assert_eq!(
        netdev_err(
            "dgram,id=d,local.type=inet,local.host=127.0.0.1,local.port=1,remote.type=unix,remote.path=/x"
        ),
        "remote and local types must be the same"
    );
    assert_eq!(
        netdev_err(
            "dgram,id=d,local.type=fd,local.str=3,remote.type=inet,remote.host=127.0.0.1,remote.port=1"
        ),
        "don't set remote with local.fd"
    );
    assert_eq!(
        netdev_err(
            "dgram,id=d,local.type=unix,local.path=/tmp/a,remote.type=inet,remote.host=230.0.0.1,remote.port=1234"
        ),
        "only support inet or fd type for local"
    );
}
