// SPDX-License-Identifier: GPL-2.0-or-later

//! Frames going through the socket, stream and dgram backends over localhost, with the
//! `info network` lines tests/qtest/netdev-socket.c looks for.

#![cfg(unix)]

mod common;

use std::os::fd::IntoRawFd;
use std::os::unix::net::{UnixDatagram, UnixStream};
use std::sync::Arc;

use common::{Recorder, attach_nic, frame, free_port, free_udp_port, netdev, tmpdir, wait_until};
use ruvm_net::{Net, Nic};

/// The `info network` line of backend `id`, as the qtest sees it, without the newline.
fn state(net: &Net, id: &str) -> String {
    let prefix = format!("{id}: index=");
    net.info_network()
        .lines()
        .map(|l| l.trim_start_matches(" \\ "))
        .find(|l| l.starts_with(&prefix))
        .unwrap_or_default()
        .to_string()
}

fn wait_state(net: &Net, id: &str, want: &str) {
    wait_until(&format!("{id} to be '{want}', it is '{}'", state(net, id)), || {
        state(net, id) == want
    });
}

fn wait_link_up(net: &Net, id: &str) {
    let nc = net.find_netdev(id).unwrap();
    wait_until(&format!("{id} link up"), || !nc.link_down());
}

/// Sends frames both ways between two NICs and checks they arrive whole and in order.
fn exchange(a: &(Arc<Nic>, Arc<Recorder>), b: &(Arc<Nic>, Arc<Recorder>)) {
    exchange_sizes(a, b, &[60, 64, 1514, 9000]);
}

/// `exchange()` with the frame sizes given. A frame the socket could not take all of at once
/// is queued (0) and finishes when the socket drains.
fn exchange_sizes(a: &(Arc<Nic>, Arc<Recorder>), b: &(Arc<Nic>, Arc<Recorder>), sizes: &[usize]) {
    for (i, &n) in sizes.iter().enumerate() {
        let r = a.0.queue().send_packet(&frame(i as u8, n));
        assert!(r == n as isize || r == 0, "sending {n} bytes gave {r}");
    }
    let got = b.1.wait_for(sizes.len());
    for (i, &n) in sizes.iter().enumerate() {
        assert_eq!(got[i], frame(i as u8, n), "frame {i}");
    }
    assert_eq!(b.0.queue().send_packet(&frame(200, 100)), 100);
    assert_eq!(a.1.wait_for(1), vec![frame(200, 100)]);
}

#[test]
fn socket_listen_connect() {
    let port = free_port();
    let mut net = Net::new();
    netdev(&mut net, &format!("socket,id=srv,listen=127.0.0.1:{port}")).unwrap();
    assert_eq!(state(&net, "srv"), "srv: index=0,type=socket,");
    netdev(&mut net, &format!("socket,id=cli,connect=127.0.0.1:{port}")).unwrap();
    assert_eq!(
        state(&net, "cli"),
        format!("cli: index=0,type=socket,socket: connect to 127.0.0.1:{port}")
    );
    let a = attach_nic(&mut net, "srv");
    let b = attach_nic(&mut net, "cli");
    wait_until("the server to accept", || {
        state(&net, "srv").contains("socket: connection from 127.0.0.1:")
    });
    wait_link_up(&net, "srv");
    wait_link_up(&net, "cli");
    exchange(&a, &b);
}

#[test]
fn socket_connection_lost_and_accepted_again() {
    let port = free_port();
    let mut net = Net::new();
    netdev(&mut net, &format!("socket,id=srv,listen=127.0.0.1:{port}")).unwrap();
    let (_nic, rec) = attach_nic(&mut net, "srv");
    for round in 0..2u8 {
        let mut peer = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        wait_until("accept", || state(&net, "srv").contains("connection from"));
        // Two frames in one write, the second split from its length.
        let mut buf = Vec::new();
        for n in [70usize, 80] {
            buf.extend_from_slice(&(n as u32).to_be_bytes());
            buf.extend_from_slice(&frame(round, n));
        }
        use std::io::Write;
        peer.write_all(&buf[..90]).unwrap();
        peer.flush().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        peer.write_all(&buf[90..]).unwrap();
        assert_eq!(rec.wait_for(2), vec![frame(round, 70), frame(round, 80)]);
        drop(peer);
        wait_state(&net, "srv", "srv: index=0,type=socket,");
    }
}

#[test]
fn socket_udp() {
    let (p0, p1) = (free_udp_port(), free_udp_port());
    let mut net = Net::new();
    netdev(&mut net, &format!("socket,id=u0,udp=127.0.0.1:{p1},localaddr=127.0.0.1:{p0}")).unwrap();
    netdev(&mut net, &format!("socket,id=u1,udp=127.0.0.1:{p0},localaddr=127.0.0.1:{p1}")).unwrap();
    assert_eq!(state(&net, "u0"), format!("u0: index=0,type=socket,socket: udp=127.0.0.1:{p1}"));
    let a = attach_nic(&mut net, "u0");
    let b = attach_nic(&mut net, "u1");
    exchange(&a, &b);
}

#[test]
fn socket_fd_stream_and_dgram() {
    let (s0, s1) = UnixStream::pair().unwrap();
    let raw = s0.into_raw_fd();
    let mut net = Net::new();
    netdev(&mut net, &format!("socket,id=fs,fd={raw}")).unwrap();
    assert_eq!(state(&net, "fs"), format!("fs: index=0,type=socket,socket: fd={raw}"));
    let (nic, rec) = attach_nic(&mut net, "fs");
    nic.queue().send_packet(&frame(1, 64));
    let mut hdr = [0u8; 4 + 64];
    use std::io::{Read, Write};
    let mut s1 = s1;
    s1.read_exact(&mut hdr).unwrap();
    assert_eq!(&hdr[..4], &64u32.to_be_bytes());
    assert_eq!(&hdr[4..], &frame(1, 64)[..]);
    s1.write_all(&[0, 0, 0, 3, 7, 8, 9]).unwrap();
    assert_eq!(rec.wait_for(1), vec![vec![7, 8, 9]]);

    let (d0, d1) = UnixDatagram::pair().unwrap();
    let raw = d0.into_raw_fd();
    netdev(&mut net, &format!("socket,id=fd,fd={raw}")).unwrap();
    assert_eq!(state(&net, "fd"), format!("fd: index=0,type=socket,socket: fd={raw} unix"));
    let (nic, rec) = attach_nic(&mut net, "fd");
    nic.queue().send_packet(&frame(2, 64));
    let mut buf = [0u8; 256];
    let n = d1.recv(&mut buf).unwrap();
    assert_eq!(&buf[..n], &frame(2, 64)[..]);
    d1.send(&frame(3, 99)).unwrap();
    assert_eq!(rec.wait_for(1), vec![frame(3, 99)]);
}

#[test]
fn socket_fd_must_be_a_socket() {
    let f = std::fs::File::open("/dev/null").unwrap();
    let raw = f.into_raw_fd();
    let e = common::netdev_err(&format!("socket,id=s,fd={raw}"));
    assert_eq!(e, "can't get socket option SO_TYPE");
}

#[test]
fn socket_mcast() {
    let port = free_udp_port();
    let mut net = Net::new();
    let opts = |id: &str| format!("socket,id={id},mcast=230.0.0.1:{port},localaddr=127.0.0.1");
    if let Err(e) = netdev(&mut net, &opts("m0")) {
        eprintln!("skipping, no multicast here: {e}");
        return;
    }
    netdev(&mut net, &opts("m1")).unwrap();
    assert_eq!(
        state(&net, "m0"),
        format!("m0: index=0,type=socket,socket: mcast=230.0.0.1:{port}")
    );
    let a = attach_nic(&mut net, "m0");
    let b = attach_nic(&mut net, "m1");
    let r = a.0.queue().send_packet(&frame(4, 64));
    if r < 0 {
        // macOS will not send from a socket bound to a group address, which is how QEMU
        // sets these up; QEMU skips its multicast test there too.
        eprintln!("skipping, multicast send failed: {r}");
        return;
    }
    // The group loops frames back, so the sender sees its own frame as well.
    assert!(b.1.wait_for(1).contains(&frame(4, 64)));
}

#[test]
fn stream_inet() {
    let port = free_port();
    let mut net = Net::new();
    netdev(
        &mut net,
        &format!("stream,id=st0,server=true,addr.type=inet,addr.host=127.0.0.1,addr.port={port}"),
    )
    .unwrap();
    assert_eq!(state(&net, "st0"), "st0: index=0,type=stream,listening");
    let mut net1 = Net::new();
    netdev(
        &mut net1,
        &format!("stream,server=false,id=st0,addr.type=inet,addr.host=127.0.0.1,addr.port={port}"),
    )
    .unwrap();
    wait_state(&net1, "st0", &format!("st0: index=0,type=stream,tcp:127.0.0.1:{port}"));
    wait_until("server to accept", || {
        state(&net, "st0").starts_with("st0: index=0,type=stream,tcp:127.0.0.1:")
    });
    let a = attach_nic(&mut net, "st0");
    let b = attach_nic(&mut net1, "st0");
    wait_link_up(&net, "st0");
    wait_link_up(&net1, "st0");
    exchange(&a, &b);

    // The server goes back to listening when the client goes away. A backend whose NIC is
    // still there is only marked deleted, so the NIC has to go as well.
    net1.cleanup();
    net1.del_nic(&b.0);
    wait_state(&net, "st0", "st0: index=0,type=stream,listening");
    assert!(net.find_netdev("st0").unwrap().link_down());
}

#[test]
fn stream_unix() {
    let dir = tmpdir("su");
    let path = dir.join("sock");
    let path = path.to_str().unwrap();
    let mut net = Net::new();
    netdev(&mut net, &format!("stream,id=st0,server=true,addr.type=unix,addr.path={path}"))
        .unwrap();
    assert_eq!(state(&net, "st0"), "st0: index=0,type=stream,listening");
    let mut net1 = Net::new();
    netdev(&mut net1, &format!("stream,id=st0,server=false,addr.type=unix,addr.path={path}"))
        .unwrap();
    let want = format!("st0: index=0,type=stream,unix:{path}");
    wait_state(&net1, "st0", &want);
    wait_state(&net, "st0", &want);
    let a = attach_nic(&mut net, "st0");
    let b = attach_nic(&mut net1, "st0");
    exchange(&a, &b);
    net.cleanup();
    net.del_nic(&a.0);
    net1.cleanup();
    net1.del_nic(&b.0);
    // The server removes its socket file when it goes.
    assert!(!dir.join("sock").exists());
    let _ = std::fs::remove_dir_all(&dir);
}

/// A Linux abstract name: nothing appears in the file system, and the connect event reports
/// the name with `abstract` and `tight` as QEMU 11.1 does, for a tight name and a padded one.
#[cfg(target_os = "linux")]
#[test]
fn stream_unix_abstract() {
    use std::sync::Mutex;

    use ruvm_net::NetEvent;
    use ruvm_qapi::types::SocketAddressU;

    for tight in [true, false] {
        let name = format!("ruvm-net-test-{}-{tight}", std::process::id());
        let addr = format!("addr.type=unix,addr.path={name},addr.abstract=on,addr.tight={tight}");
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut net = Net::new();
        let ev = Arc::clone(&events);
        net.set_event_sink(Some(Arc::new(move |e| ev.lock().unwrap().push(e))));
        netdev(&mut net, &format!("stream,id=st0,server=true,{addr}")).unwrap();
        assert!(!std::path::Path::new(&name).exists());
        let mut net1 = Net::new();
        netdev(&mut net1, &format!("stream,id=st0,server=false,{addr}")).unwrap();
        let want = format!("st0: index=0,type=stream,unix:{name}");
        wait_state(&net1, "st0", &want);
        wait_state(&net, "st0", &want);
        let a = attach_nic(&mut net, "st0");
        let b = attach_nic(&mut net1, "st0");
        exchange(&a, &b);
        let got = events.lock().unwrap().clone();
        let [NetEvent::StreamConnected { netdev_id, addr }] = &got[..] else {
            panic!("{got:?}");
        };
        assert_eq!(netdev_id, "st0");
        let SocketAddressU::Unix(u) = &addr.u else { panic!("{addr:?}") };
        assert_eq!(
            (u.path.as_str(), u.abstract_, u.tight),
            (name.as_str(), Some(true), Some(tight))
        );
        net.cleanup();
        net.del_nic(&a.0);
        net1.cleanup();
        net1.del_nic(&b.0);
    }
}

#[test]
fn stream_unix_reconnect() {
    let dir = tmpdir("sr");
    let path = dir.join("sock");
    let path = path.to_str().unwrap();
    let server = format!("stream,id=st0,server=true,addr.type=unix,addr.path={path}");
    let mut net0 = Net::new();
    netdev(&mut net0, &server).unwrap();
    let mut net1 = Net::new();
    netdev(
        &mut net1,
        &format!("stream,server=false,id=st0,addr.type=unix,addr.path={path},reconnect-ms=100"),
    )
    .unwrap();
    let want = format!("st0: index=0,type=stream,unix:{path}");
    wait_state(&net0, "st0", &want);
    wait_state(&net1, "st0", &want);

    // Kill the server; the client notices and tries again.
    net0.cleanup();
    let nc = net1.find_netdev("st0").unwrap();
    wait_until("client to notice", || nc.link_down());

    let mut net0 = Net::new();
    netdev(&mut net0, &server).unwrap();
    wait_state(&net0, "st0", &want);
    wait_state(&net1, "st0", &want);
    let a = attach_nic(&mut net0, "st0");
    let b = attach_nic(&mut net1, "st0");
    wait_link_up(&net1, "st0");
    exchange(&a, &b);
    net1.cleanup();
    net0.cleanup();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn stream_connect_failure_shows_in_info() {
    let dir = tmpdir("sf");
    let path = dir.join("none");
    let mut net = Net::new();
    netdev(
        &mut net,
        &format!("stream,id=st0,server=off,addr.type=unix,addr.path={}", path.display()),
    )
    .unwrap();
    wait_until("the error", || state(&net, "st0").starts_with("st0: index=0,type=stream,error: "));
    assert!(net.find_netdev("st0").unwrap().link_down());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn stream_fd() {
    let (s0, s1) = UnixStream::pair().unwrap();
    let (r0, r1) = (s0.into_raw_fd(), s1.into_raw_fd());
    let mut net0 = Net::new();
    netdev(&mut net0, &format!("stream,id=st0,addr.type=fd,addr.str={r0}")).unwrap();
    wait_state(&net0, "st0", "st0: index=0,type=stream,unix:");
    let mut net1 = Net::new();
    netdev(&mut net1, &format!("stream,id=st0,addr.type=fd,addr.str={r1}")).unwrap();
    wait_state(&net1, "st0", "st0: index=0,type=stream,unix:");
    let a = attach_nic(&mut net0, "st0");
    let b = attach_nic(&mut net1, "st0");
    exchange(&a, &b);
}

#[test]
fn stream_fd_server() {
    let dir = tmpdir("sfs");
    let path = dir.join("sock");
    let listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
    let raw = listener.into_raw_fd();
    let mut net = Net::new();
    netdev(&mut net, &format!("stream,id=st0,server=on,addr.type=fd,addr.str={raw}")).unwrap();
    assert_eq!(state(&net, "st0"), "st0: index=0,type=stream,listening");
    let (_nic, rec) = attach_nic(&mut net, "st0");
    let mut c = UnixStream::connect(&path).unwrap();
    use std::io::Write;
    c.write_all(&[0, 0, 0, 2, 0xaa, 0xbb]).unwrap();
    assert_eq!(rec.wait_for(1), vec![vec![0xaa, 0xbb]]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn dgram_inet() {
    let (p0, p1) = (free_udp_port(), free_udp_port());
    let mut net0 = Net::new();
    netdev(
        &mut net0,
        &format!(
            "dgram,id=st0,local.type=inet,local.host=127.0.0.1,local.port={p0},\
             remote.type=inet,remote.host=127.0.0.1,remote.port={p1}"
        ),
    )
    .unwrap();
    assert_eq!(
        state(&net0, "st0"),
        format!("st0: index=0,type=dgram,udp=127.0.0.1:{p0}/127.0.0.1:{p1}")
    );
    let mut net1 = Net::new();
    netdev(
        &mut net1,
        &format!(
            "dgram,id=st0,local.type=inet,local.host=127.0.0.1,local.port={p1},\
             remote.type=inet,remote.host=127.0.0.1,remote.port={p0}"
        ),
    )
    .unwrap();
    assert_eq!(
        state(&net1, "st0"),
        format!("st0: index=0,type=dgram,udp=127.0.0.1:{p1}/127.0.0.1:{p0}")
    );
    let a = attach_nic(&mut net0, "st0");
    let b = attach_nic(&mut net1, "st0");
    exchange(&a, &b);
}

#[test]
fn dgram_unix() {
    let dir = tmpdir("du");
    let (l0, l1) = (dir.join("a"), dir.join("b"));
    let (l0, l1) = (l0.to_str().unwrap(), l1.to_str().unwrap());
    let mut net0 = Net::new();
    netdev(
        &mut net0,
        &format!("dgram,id=st0,local.type=unix,local.path={l0},remote.type=unix,remote.path={l1}"),
    )
    .unwrap();
    assert_eq!(state(&net0, "st0"), format!("st0: index=0,type=dgram,udp={l0}:{l1}"));
    let mut net1 = Net::new();
    netdev(
        &mut net1,
        &format!("dgram,id=st0,local.type=unix,local.path={l1},remote.type=unix,remote.path={l0}"),
    )
    .unwrap();
    assert_eq!(state(&net1, "st0"), format!("st0: index=0,type=dgram,udp={l1}:{l0}"));
    let a = attach_nic(&mut net0, "st0");
    let b = attach_nic(&mut net1, "st0");
    // macOS caps Unix datagrams at 2048 bytes unless told otherwise.
    exchange_sizes(&a, &b, &[60, 64, 1514]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn dgram_fd() {
    let (d0, d1) = UnixDatagram::pair().unwrap();
    let (r0, r1) = (d0.into_raw_fd(), d1.into_raw_fd());
    let mut net0 = Net::new();
    netdev(&mut net0, &format!("dgram,id=st0,local.type=fd,local.str={r0}")).unwrap();
    assert_eq!(state(&net0, "st0"), format!("st0: index=0,type=dgram,fd={r0} unix"));
    let mut net1 = Net::new();
    netdev(&mut net1, &format!("dgram,id=st0,local.type=fd,local.str={r1}")).unwrap();
    assert_eq!(state(&net1, "st0"), format!("st0: index=0,type=dgram,fd={r1} unix"));
    let a = attach_nic(&mut net0, "st0");
    let b = attach_nic(&mut net1, "st0");
    exchange_sizes(&a, &b, &[60, 64, 1514]);
}

#[test]
fn dgram_mcast() {
    let mut net = Net::new();
    match netdev(&mut net, "dgram,id=st0,remote.type=inet,remote.host=230.0.0.1,remote.port=1234") {
        Ok(()) => {
            assert_eq!(state(&net, "st0"), "st0: index=0,type=dgram,mcast=230.0.0.1:1234");
            assert_eq!(net.find_netdev("st0").unwrap().model(), "dram");
        }
        // QEMU skips this test on macOS too, where joining may fail without a route.
        Err(e) => eprintln!("skipping, no multicast here: {e}"),
    }
}

#[test]
fn legacy_socket_through_hub() {
    let (p0, p1) = (free_udp_port(), free_udp_port());
    let mut net = Net::new();
    net.parse_net("nic,model=e1000").unwrap();
    net.parse_net(&format!("socket,udp=127.0.0.1:{p1},localaddr=127.0.0.1:{p0}")).unwrap();
    net.init_clients().unwrap();
    let port = net.nd_table()[0].netdev.clone().unwrap();
    // The NIC the machine would make, on the hub port the -net nic got.
    let rec = Recorder::new();
    let r = rec.clone();
    let conf = ruvm_net::NicConf { macaddr: net.nd_table()[0].macaddr, peers: vec![port] };
    let nic = net.new_nic(&conf, "e1000", None, move |_, _| r.clone());
    let peer = std::net::UdpSocket::bind(("127.0.0.1", p1)).unwrap();
    peer.set_read_timeout(Some(std::time::Duration::from_secs(10))).unwrap();
    nic.queue().send_packet(&frame(1, 64));
    let mut buf = [0u8; 2048];
    let n = peer.recv(&mut buf).unwrap();
    assert_eq!(&buf[..n], &frame(1, 64)[..]);
    peer.send_to(&frame(2, 64), ("127.0.0.1", p0)).unwrap();
    assert_eq!(rec.wait_for(1), vec![frame(2, 64)]);
    let info = net.info_network();
    // Ports are listed newest first, as QEMU does.
    let lines: Vec<&str> = info.lines().collect();
    assert_eq!(lines[0], "hub 0", "{info}");
    assert!(lines[1].starts_with(" \\ hub0port1: #net"), "{info}");
    assert!(lines[1].ends_with(&format!(",type=socket,socket: udp=127.0.0.1:{p1}")), "{info}");
    assert_eq!(lines[2], " \\ hub0port0: e1000.0: index=0,type=nic,", "{info}");
}
