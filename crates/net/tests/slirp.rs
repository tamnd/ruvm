// SPDX-License-Identifier: GPL-2.0-or-later

//! `-netdev user`: the option checks, which need no library, and a live stack answering DHCP
//! and ARP, with host forwarding rules added and removed. The live tests are skipped when
//! libslirp cannot be loaded.

#![cfg(all(unix, feature = "slirp"))]

mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use common::{Recorder, attach_nic, free_port, netdev, netdev_err};
use ruvm_net::{Net, libslirp_version};

const GUEST_MAC: [u8; 6] = [0x52, 0x54, 0, 0x12, 0x34, 0x56];

fn have_libslirp() -> bool {
    match libslirp_version() {
        Ok(_) => true,
        Err(e) => {
            eprintln!("skipping: libslirp could not be loaded: {e}");
            false
        }
    }
}

#[test]
fn option_errors() {
    let cases = [
        ("user,id=u,ipv4=off,net=10.0.0.0/8", "IPv4 disabled but netmask/host/dns provided"),
        ("user,id=u,ipv4=off,dns=10.0.2.3", "IPv4 disabled but netmask/host/dns provided"),
        ("user,id=u,ipv6=off,ipv6-host=fec0::2", "IPv6 disabled but prefix/host6/dns6 provided"),
        ("user,id=u,ipv4=off,ipv6=off", "IPv4 and IPv6 disabled"),
        ("user,id=u,ipv4=on,ipv6-dns=fec0::3", "IPv6 disabled but prefix/host6/dns6 provided"),
        ("user,id=u,net=nonsense", "Failed to parse netmask"),
        ("user,id=u,net=nonsense/24", "Failed to parse netmask"),
        ("user,id=u,net=10.0.2.0/33", "Invalid netmask provided (must be in range 4-32)"),
        ("user,id=u,net=10.0.2.0/3", "Invalid netmask provided (must be in range 4-32)"),
        ("user,id=u,net=10.0.2.0/255.x", "Failed to parse netmask (trailing chars)"),
        ("user,id=u,ip=nonsense", "Failed to parse netmask"),
        ("user,id=u,host=x", "Failed to parse host"),
        ("user,id=u,host=10.0.3.2", "Host doesn't belong to network"),
        ("user,id=u,dns=x", "Failed to parse DNS"),
        ("user,id=u,restrict=on,dns=10.0.3.3", "DNS doesn't belong to network"),
        ("user,id=u,dns=10.0.2.2", "DNS must be different from host"),
        ("user,id=u,dhcpstart=x", "Failed to parse DHCP start address"),
        ("user,id=u,dhcpstart=10.0.3.15", "DHCP doesn't belong to network"),
        ("user,id=u,dhcpstart=10.0.2.3", "DHCP must be different from host and DNS"),
        ("user,id=u,smbserver=x", "Failed to parse SMB address"),
        ("user,id=u,ipv6-prefix=x", "Failed to parse IPv6 prefix"),
        (
            "user,id=u,ipv6-prefixlen=127",
            "Invalid IPv6 prefix provided (IPv6 prefix length must be between 0 and 126)",
        ),
        (
            "user,id=u,ipv6-net=fec0::/200",
            "Invalid IPv6 prefix provided (IPv6 prefix length must be between 0 and 126)",
        ),
        ("user,id=u,ipv6-net=fec0::/x", "parameter 'ipv6-net' expects a number after '/'"),
        ("user,id=u,ipv6-host=x", "Failed to parse IPv6 host"),
        ("user,id=u,ipv6-host=fec1::2", "IPv6 Host doesn't belong to network"),
        ("user,id=u,ipv6-dns=x", "Failed to parse IPv6 DNS"),
        ("user,id=u,restrict=on,ipv6-dns=fec1::3", "IPv6 DNS doesn't belong to network"),
        ("user,id=u,domainname=", "'domainname' parameter cannot be empty"),
    ];
    for (opts, want) in cases {
        assert_eq!(netdev_err(opts), want, "for {opts}");
    }
    let long = "x".repeat(256);
    assert_eq!(
        netdev_err(&format!("user,id=u,domainname={long}")),
        "'domainname' parameter cannot exceed 255 bytes"
    );
    assert_eq!(
        netdev_err(&format!("user,id=u,hostname={long}")),
        "'vhostname' parameter cannot exceed 255 bytes"
    );
    assert_eq!(
        netdev_err(&format!("user,id=u,tftp-server-name={long}")),
        "'tftp-server-name' parameter cannot exceed 255 bytes"
    );
}

#[test]
fn monitor_lookup_errors() {
    let mut net = Net::new();
    assert_eq!(
        net.hostfwd_add(None, "tcp::1-:2").unwrap_err().message(),
        "user mode network stack not in use"
    );
    assert_eq!(
        net.hostfwd_remove(Some("nope"), "tcp::1").unwrap_err().message(),
        "unrecognized netdev id 'nope'"
    );
    netdev(&mut net, "hubport,id=h,hubid=0").unwrap();
    assert_eq!(
        net.hostfwd_add(Some("h"), "tcp::1-:2").unwrap_err().message(),
        "invalid device specified"
    );
    assert_eq!(net.info_usernet(), "");
}

/// The one's complement sum of RFC 1071.
fn checksum(data: &[u8]) -> u16 {
    let mut sum = 0u32;
    for c in data.chunks(2) {
        sum += u32::from(u16::from_be_bytes([c[0], *c.get(1).unwrap_or(&0)]));
    }
    while sum > 0xffff {
        sum = (sum & 0xffff) + (sum >> 16);
    }
    !(sum as u16)
}

/// An Ethernet frame with a UDP datagram inside, from 0.0.0.0 to the broadcast address.
fn udp_broadcast(sport: u16, dport: u16, payload: &[u8]) -> Vec<u8> {
    let mut f = Vec::new();
    f.extend_from_slice(&[0xff; 6]);
    f.extend_from_slice(&GUEST_MAC);
    f.extend_from_slice(&[0x08, 0x00]);
    let total = (20 + 8 + payload.len()) as u16;
    let mut ip = vec![0x45, 0, 0, 0, 0, 1, 0, 0, 64, 17, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255];
    ip[2..4].copy_from_slice(&total.to_be_bytes());
    let c = checksum(&ip);
    ip[10..12].copy_from_slice(&c.to_be_bytes());
    f.extend_from_slice(&ip);
    f.extend_from_slice(&sport.to_be_bytes());
    f.extend_from_slice(&dport.to_be_bytes());
    f.extend_from_slice(&((8 + payload.len()) as u16).to_be_bytes());
    f.extend_from_slice(&[0, 0]);
    f.extend_from_slice(payload);
    f
}

fn dhcp_discover(xid: u32) -> Vec<u8> {
    let mut b = vec![0u8; 236];
    b[0] = 1; // BOOTREQUEST
    b[1] = 1; // Ethernet
    b[2] = 6;
    b[4..8].copy_from_slice(&xid.to_be_bytes());
    b[28..34].copy_from_slice(&GUEST_MAC);
    b.extend_from_slice(&[99, 130, 83, 99]);
    b.extend_from_slice(&[53, 1, 1]); // DHCPDISCOVER
    b.push(255);
    b.resize(300, 0);
    udp_broadcast(68, 67, &b)
}

/// DHCP options as (code, value).
type DhcpOptions = Vec<(u8, Vec<u8>)>;

/// The DHCP options of a BOOTP reply inside an Ethernet frame, if that is what `f` is.
fn dhcp_reply(f: &[u8], xid: u32) -> Option<([u8; 4], DhcpOptions)> {
    if f.len() < 14 + 20 + 8 + 240 || f[12..14] != [0x08, 0x00] || f[14 + 9] != 17 {
        return None;
    }
    let ihl = usize::from(f[14] & 0xf) * 4;
    let udp = &f[14 + ihl..];
    if udp[0..2] != 67u16.to_be_bytes() || udp[2..4] != 68u16.to_be_bytes() {
        return None;
    }
    let b = &udp[8..];
    if b[0] != 2 || b[4..8] != xid.to_be_bytes() {
        return None;
    }
    let yiaddr = [b[16], b[17], b[18], b[19]];
    let mut opts = Vec::new();
    let mut i = 240;
    while i < b.len() && b[i] != 255 {
        if b[i] == 0 {
            i += 1;
            continue;
        }
        let len = usize::from(b[i + 1]);
        opts.push((b[i], b[i + 2..i + 2 + len].to_vec()));
        i += 2 + len;
    }
    Some((yiaddr, opts))
}

/// Waits for a frame `f` accepts, dropping the others.
fn wait_frame<T>(rec: &Arc<Recorder>, mut f: impl FnMut(&[u8]) -> Option<T>) -> T {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        for frame in rec.take() {
            if let Some(v) = f(&frame) {
                return v;
            }
        }
        assert!(Instant::now() < deadline, "timed out waiting for a frame");
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn dhcp_offer() {
    if !have_libslirp() {
        return;
    }
    let mut net = Net::new();
    netdev(&mut net, "user,id=u").unwrap();
    assert_eq!(common::info(&net, "u"), "net=10.0.2.0,restrict=off");
    let (nic, rec) = attach_nic(&mut net, "u");
    nic.queue().send_packet(&dhcp_discover(0x1234_5678));
    let (yiaddr, opts) = wait_frame(&rec, |f| dhcp_reply(f, 0x1234_5678));
    assert_eq!(yiaddr, [10, 0, 2, 15]);
    let opt = |code: u8| opts.iter().find(|(c, _)| *c == code).map(|(_, v)| v.clone());
    assert_eq!(opt(53), Some(vec![2]), "DHCPOFFER");
    assert_eq!(opt(3), Some(vec![10, 0, 2, 2]), "router");
    assert_eq!(opt(6), Some(vec![10, 0, 2, 3]), "DNS");
    assert_eq!(opt(1), Some(vec![255, 255, 255, 0]), "netmask");
}

#[test]
fn dhcp_offer_on_another_network() {
    if !have_libslirp() {
        return;
    }
    let mut net = Net::new();
    netdev(&mut net, "user,id=u,net=192.168.76.0/24,dhcpstart=192.168.76.9").unwrap();
    assert_eq!(common::info(&net, "u"), "net=192.168.76.0,restrict=off");
    let (nic, rec) = attach_nic(&mut net, "u");
    nic.queue().send_packet(&dhcp_discover(7));
    let (yiaddr, opts) = wait_frame(&rec, |f| dhcp_reply(f, 7));
    assert_eq!(yiaddr, [192, 168, 76, 9]);
    let router = opts.iter().find(|(c, _)| *c == 3).map(|(_, v)| v.clone());
    assert_eq!(router, Some(vec![192, 168, 76, 2]));
}

#[test]
fn arp_for_the_gateway() {
    if !have_libslirp() {
        return;
    }
    let mut net = Net::new();
    netdev(&mut net, "user,id=u").unwrap();
    let (nic, rec) = attach_nic(&mut net, "u");
    let mut f = Vec::new();
    f.extend_from_slice(&[0xff; 6]);
    f.extend_from_slice(&GUEST_MAC);
    f.extend_from_slice(&[0x08, 0x06, 0, 1, 0x08, 0x00, 6, 4, 0, 1]);
    f.extend_from_slice(&GUEST_MAC);
    f.extend_from_slice(&[10, 0, 2, 15]);
    f.extend_from_slice(&[0; 6]);
    f.extend_from_slice(&[10, 0, 2, 2]);
    nic.queue().send_packet(&f);
    let reply = wait_frame(&rec, |f| {
        (f.len() >= 42 && f[12..14] == [0x08, 0x06] && f[20..22] == [0, 2]).then(|| f.to_vec())
    });
    assert_eq!(reply[0..6], GUEST_MAC, "sent back to the guest");
    assert_eq!(reply[28..32], [10, 0, 2, 2], "for the gateway address");
    assert_eq!(reply[22..24], [0x52, 0x55], "from slirp's own MAC range");
    assert_eq!(reply[38..42], [10, 0, 2, 15]);
    assert!(reply.len() >= 60, "padded for the NIC");
}

#[test]
fn hostfwd_add_and_remove() {
    if !have_libslirp() {
        return;
    }
    let mut net = Net::new();
    let p1 = free_port();
    netdev(&mut net, &format!("user,id=u,hostfwd=tcp:127.0.0.1:{p1}-:22")).unwrap();
    // The rule listens right away, without a guest.
    std::net::TcpStream::connect(("127.0.0.1", p1)).unwrap();

    let p2 = free_port();
    let rule = format!("tcp:127.0.0.1:{p2}-10.0.2.15:80");
    net.hostfwd_add(Some("u"), &rule).unwrap();
    std::net::TcpStream::connect(("127.0.0.1", p2)).unwrap();
    assert_eq!(
        net.hostfwd_add(None, &rule).unwrap_err().message(),
        format!("Could not set up host forwarding rule '{rule}'")
    );
    let src = format!("tcp:127.0.0.1:{p2}");
    assert_eq!(
        net.hostfwd_remove(Some("u"), &src).unwrap(),
        format!("host forwarding rule for {src} removed")
    );
    assert_eq!(
        net.hostfwd_remove(None, &src).unwrap(),
        format!("host forwarding rule for {src} not found")
    );
    assert!(std::net::TcpStream::connect(("127.0.0.1", p2)).is_err());
    assert_eq!(net.hostfwd_remove(None, "sctp::1").unwrap(), "invalid format");
    assert_eq!(net.hostfwd_remove(None, "tcp:1").unwrap(), "invalid format");
    assert_eq!(net.hostfwd_remove(None, "tcp::1x").unwrap(), "invalid format");

    let p3 = free_port();
    net.hostfwd_add(None, &format!("udp::{p3}-:53")).unwrap();
    assert_eq!(
        net.hostfwd_remove(None, &format!("udp::{p3}")).unwrap(),
        format!("host forwarding rule for udp::{p3} removed")
    );

    let bad = [
        ("nocolon", "No : separators"),
        ("sctp::1-:2", "Bad protocol name"),
        ("tcp:1", "Missing : separator"),
        ("tcp:1-:2", "Bad host address"),
        ("tcp:x:1-:2", "Bad host address"),
        ("tcp::1:2", "Bad host port separator"),
        ("tcp::x-:2", "Bad host port"),
        ("tcp::70000-:2", "Bad host port"),
        ("tcp::1-2", "Missing guest address"),
        ("tcp::1-x:2", "Bad guest address"),
        ("tcp::1-:0", "Bad guest port"),
        ("tcp::1-:", "Bad guest port"),
        ("unix:rel/path-:22", "unix socket path must be absolute"),
        ("unix:-:22", "Missing unix socket path"),
        ("unix:/tmp/x", "Missing - separator"),
    ];
    for (rule, why) in bad {
        assert_eq!(
            net.hostfwd_add(Some("u"), rule).unwrap_err().message(),
            format!("Invalid host forwarding rule '{rule}' ({why})"),
        );
    }
    let usernet = net.info_usernet();
    assert!(usernet.starts_with("Hub -1 (u):\n"), "{usernet}");
    assert!(usernet.contains("HOST_FORWARD"), "{usernet}");
}

#[test]
fn hostfwd_unix_socket() {
    if !have_libslirp() {
        return;
    }
    let dir = common::tmpdir("usock");
    let path = dir.join("fwd.sock");
    let mut net = Net::new();
    netdev(&mut net, &format!("user,id=u,hostfwd=unix:{}-:22", path.display())).unwrap();
    std::os::unix::net::UnixStream::connect(&path).unwrap();
    let file = dir.join("plain");
    std::fs::write(&file, b"").unwrap();
    let rule = format!("unix:{}-:22", file.display());
    assert_eq!(
        net.hostfwd_add(None, &rule).unwrap_err().message(),
        format!("Invalid host forwarding rule '{rule}' (file exists and it's not unix socket)")
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn bad_rules_on_the_command_line() {
    if !have_libslirp() {
        return;
    }
    assert_eq!(
        netdev_err("user,id=u,hostfwd=tcp:bad"),
        "Invalid host forwarding rule 'tcp:bad' (Missing : separator)"
    );
    for rule in
        ["udp:10.0.2.100:80-cmd:cat", "tcp:x:80-cmd:cat", "tcp::0-cmd:cat", "tcp::8x-cmd:cat"]
    {
        assert_eq!(
            netdev_err(&format!("user,id=u,guestfwd={rule}")),
            format!("Invalid guest forwarding rule '{rule}'")
        );
    }
    assert_eq!(
        netdev_err("user,id=u,guestfwd=tcp:10.0.2.100:80-file:/dev/null"),
        "Could not open guest forwarding device 'guestfwd.tcp.80'"
    );
    // Rules are set up after the client is made; a failure removes it again.
    let mut net = Net::new();
    assert!(netdev(&mut net, "user,id=u,hostfwd=tcp::1-:0").is_err());
    assert!(net.find_netdev("u").is_none());
    netdev(&mut net, "user,id=u,guestfwd=tcp:10.0.2.100:80-cmd:cat").unwrap();
}

#[test]
fn guestfwd_to_tcp() {
    if !have_libslirp() {
        return;
    }
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let mut net = Net::new();
    netdev(&mut net, &format!("user,id=u,guestfwd=tcp:10.0.2.100:1234-tcp:127.0.0.1:{port}"))
        .unwrap();
    listener.accept().unwrap();
}

#[test]
fn user_with_net_joins_a_hub() {
    if !have_libslirp() {
        return;
    }
    let mut net = Net::new();
    net.parse_net("user,id=legacy").unwrap();
    net.init_clients().unwrap();
    let usernet = net.info_usernet();
    assert!(usernet.starts_with("Hub 0 (legacy):\n"), "{usernet}");
    net.netdev_del("legacy").unwrap_or(());
}

#[test]
fn delete_and_recreate() {
    if !have_libslirp() {
        return;
    }
    let mut net = Net::new();
    let p = free_port();
    netdev(&mut net, &format!("user,id=u,hostfwd=tcp:127.0.0.1:{p}-:22")).unwrap();
    net.netdev_del("u").unwrap();
    assert_eq!(net.info_usernet(), "");
    // The listening socket went away with the stack, so the port can be taken again.
    netdev(&mut net, &format!("user,id=u,hostfwd=tcp:127.0.0.1:{p}-:22")).unwrap();
}

#[test]
fn restrict_shows_in_info() {
    if !have_libslirp() {
        return;
    }
    let mut net = Net::new();
    netdev(&mut net, "user,id=u,restrict=on,ipv6=off,dnssearch=a.example,dnssearch=b.example")
        .unwrap();
    assert_eq!(common::info(&net, "u"), "net=10.0.2.0,restrict=on");
}
