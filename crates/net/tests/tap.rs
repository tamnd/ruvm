// SPDX-License-Identifier: GPL-2.0-or-later

//! The tap backend. Real tap devices need Linux and root; the rest uses descriptors.

#![cfg(unix)]

mod common;

use std::os::fd::IntoRawFd;
use std::os::unix::net::UnixDatagram;

use common::{attach_nic, frame, netdev, netdev_err};
use ruvm_net::Net;

/// Any descriptor passes as a tap with `fd=` where there is no TUNGETIFF to ask, so a
/// datagram socket pair stands in for the device.
#[cfg(not(target_os = "linux"))]
#[test]
fn tap_fd_moves_frames() {
    let (d0, d1) = UnixDatagram::pair().unwrap();
    let raw = d0.into_raw_fd();
    let mut net = Net::new();
    netdev(&mut net, &format!("tap,id=t0,fd={raw},vhost=on")).unwrap();
    let nc = net.find_netdev("t0").unwrap();
    assert_eq!(nc.info_str(), format!("fd={raw}"));
    assert_eq!(nc.model(), "tap");
    assert!(!nc.has_vnet_hdr());
    let (nic, rec) = attach_nic(&mut net, "t0");

    nic.queue().send_packet(&frame(1, 64));
    let mut buf = [0u8; 256];
    let n = d1.recv(&mut buf).unwrap();
    assert_eq!(&buf[..n], &frame(1, 64)[..]);

    // Short frames from the host get padded for the NIC.
    d1.send(b"tiny").unwrap();
    let got = rec.wait_for(1);
    assert_eq!(got[0].len(), 60);
    assert_eq!(&got[0][..4], b"tiny");

    d1.send(&frame(2, 1514)).unwrap();
    assert_eq!(rec.wait_for(1), vec![frame(2, 1514)]);

    net.cleanup();
    net.del_nic(&nic);
    // The backend owned the descriptor and closed it, so the other end sees nobody.
    assert!(d1.send(b"x").is_err());
}

#[cfg(not(target_os = "linux"))]
#[test]
fn tap_fds_make_queues() {
    let (a0, _a1) = UnixDatagram::pair().unwrap();
    let (b0, _b1) = UnixDatagram::pair().unwrap();
    let (ra, rb) = (a0.into_raw_fd(), b0.into_raw_fd());
    let mut net = Net::new();
    netdev(&mut net, &format!("tap,id=mq,fds={ra}:{rb}")).unwrap();
    let queues: Vec<_> = net.clients().iter().filter(|c| c.name() == "mq").cloned().collect();
    assert_eq!(queues.len(), 2);
    assert_eq!(queues[0].info_str(), format!("fd={ra}"));
    assert_eq!(queues[1].info_str(), format!("fd={rb}"));
    net.cleanup();
}

#[cfg(target_os = "linux")]
#[test]
fn tap_fd_must_be_a_tap() {
    let (d0, _d1) = UnixDatagram::pair().unwrap();
    let raw = d0.into_raw_fd();
    let e = netdev_err(&format!("tap,id=t0,fd={raw}"));
    assert!(e.starts_with(&format!("Unable to query TUNGETIFF on FD {raw}: ")), "{e}");
}

#[test]
fn tap_fd_must_exist() {
    let e = netdev_err("tap,id=t0,fd=bogus");
    assert_eq!(e, "Invalid file descriptor number 'bogus'");
}

/// Whether this process may make tap devices.
#[cfg(target_os = "linux")]
fn can_make_taps() -> bool {
    use std::os::unix::fs::MetadataExt;
    let root = std::fs::metadata("/proc/self").is_ok_and(|m| m.uid() == 0);
    root && std::path::Path::new("/dev/net/tun").exists()
}

#[cfg(target_os = "linux")]
#[test]
fn tap_device_as_root() {
    if !can_make_taps() {
        eprintln!("skipping, needs root and /dev/net/tun");
        return;
    }
    let ifname = format!("rvt{}", std::process::id() % 100_000);
    let mut net = Net::new();
    if let Err(e) = netdev(&mut net, &format!("tap,id=t0,ifname={ifname},script=no,downscript=no"))
    {
        eprintln!("skipping, no tap here: {e}");
        return;
    }
    let nc = net.find_netdev("t0").unwrap();
    assert_eq!(nc.info_str(), format!("ifname={ifname},script=no,downscript=no"));
    assert!(nc.has_vnet_hdr());
    nc.set_vnet_hdr_len(ruvm_net::VNET_HDR_MRG_RXBUF_LEN);
    assert_eq!(nc.vnet_hdr_len(), ruvm_net::VNET_HDR_MRG_RXBUF_LEN);
    let (nic, _rec) = attach_nic(&mut net, "t0");
    // The device is down, so the frame goes nowhere, but the write has to work.
    let mut f = vec![0u8; ruvm_net::VNET_HDR_MRG_RXBUF_LEN];
    f.extend_from_slice(&frame(1, 64));
    assert!(nic.queue().send_packet(&f) >= 0);
    net.cleanup();
    net.del_nic(&nic);

    // Two queues on one device.
    let mut net = Net::new();
    netdev(&mut net, &format!("tap,id=mq,ifname={ifname},queues=2,script=no,downscript=no"))
        .unwrap();
    assert_eq!(net.clients().iter().filter(|c| c.name() == "mq").count(), 2);
    net.cleanup();
}

#[cfg(target_os = "linux")]
#[test]
fn tap_script_failure() {
    if !can_make_taps() {
        eprintln!("skipping, needs root and /dev/net/tun");
        return;
    }
    let e = netdev_err("tap,id=t0,script=/bin/false,downscript=no");
    assert!(e.starts_with("network script /bin/false failed with status "), "{e}");
}
