// SPDX-License-Identifier: GPL-2.0-or-later

//! The old socket backend, net/socket.c: `listen=`, `connect=`, `mcast=`, `udp=` and `fd=`.
//!
//! TCP connections carry frames with a 4 byte length in front, UDP carries one frame per
//! datagram. Only IPv4 is supported, as in QEMU.

use std::sync::Arc;

use rustix::io::Errno;
use rustix::net::{AddressFamily, SocketAddrAny, SocketType};
use ruvm_base::{Error, Result};
use ruvm_qapi::types::{NetClientDriver, Netdev, NetdevU};

use crate::client::NetClient;
use crate::net::Net;
use crate::poll::unblock;
use crate::sock::{
    Flavour, Framing, SockConfig, address_type_str, local_address, mcast_create, new_sock,
    new_socket, udp_bind,
};
use crate::util::{inet_aton, parse_host_port};

const MODEL: &str = "socket";

/// `net_socket_fd_check()`.
fn fd_check(fd: &std::os::fd::OwnedFd, raw: i32) -> Result<SocketType> {
    let ty = rustix::net::sockopt::socket_type(fd)
        .map_err(|_| Error::generic("can't get socket option SO_TYPE"))?;
    if ty != SocketType::DGRAM && ty != SocketType::STREAM {
        return Err(Error::generic(format!(
            "socket type={} for fd={raw} must be either SOCK_DGRAM or SOCK_STREAM",
            ty.as_raw()
        )));
    }
    Ok(ty)
}

/// `net_init_socket()`.
pub(crate) fn net_init_socket(
    net: &mut Net,
    netdev: &Netdev,
    name: &str,
    peer: Option<Arc<NetClient>>,
) -> Result<()> {
    let NetdevU::Socket(sock) = &netdev.u else {
        unreachable!("socket init with another type");
    };
    let peer = peer.as_ref();
    let count = [&sock.fd, &sock.listen, &sock.connect, &sock.mcast, &sock.udp]
        .iter()
        .filter(|o| o.is_some())
        .count();
    if count != 1 {
        return Err(Error::generic("exactly one of listen=, connect=, mcast= or udp= is required"));
    }
    if sock.localaddr.is_some() && sock.mcast.is_none() && sock.udp.is_none() {
        return Err(Error::generic("localaddr= is only valid with mcast= or udp="));
    }

    if let Some(fdstr) = &sock.fd {
        let raw = net.fd_param(fdstr)?;
        let fd =
            crate::fd::adopt(raw).map_err(|_| Error::generic("can't get socket option SO_TYPE"))?;
        let ty = fd_check(&fd, raw)?;
        unblock(&fd)?;
        if ty == SocketType::DGRAM {
            // net_socket_fd_init_dgram() with is_connected set but no mcast=, which the
            // check above rules out.
            let sa = local_address(&fd)?;
            let mut cfg = SockConfig::new(Framing::Dgram, Flavour::Socket);
            cfg.info = format!("socket: fd={raw} {}", address_type_str(&sa));
            cfg.fd = Some(fd);
            new_sock(net, NetClientDriver::Socket, peer, MODEL, name, cfg)?;
        } else {
            // net_socket_fd_init_stream()
            let _ = rustix::net::sockopt::set_tcp_nodelay(&fd, true);
            let mut cfg = SockConfig::new(Framing::Stream, Flavour::Socket);
            cfg.info = format!("socket: fd={raw}");
            cfg.fd = Some(fd);
            new_sock(net, NetClientDriver::Socket, peer, MODEL, name, cfg)?;
        }
        return Ok(());
    }

    if let Some(host) = &sock.listen {
        // net_socket_listen_init()
        let saddr = parse_host_port(host)?;
        let fd = new_socket(AddressFamily::INET, SocketType::STREAM)
            .map_err(|e| Error::from_io("can't create stream socket", e))?;
        unblock(&fd)?;
        let _ = rustix::net::sockopt::set_socket_reuseaddr(&fd, true);
        rustix::net::bind(&fd, &saddr).map_err(|e| {
            Error::from_io(format!("can't bind ip={} to socket", saddr.ip()), e.into())
        })?;
        rustix::net::listen(&fd, 0)
            .map_err(|e| Error::from_io("can't listen on socket", e.into()))?;
        let mut cfg = SockConfig::new(Framing::Stream, Flavour::Socket);
        cfg.listen_fd = Some(fd);
        new_sock(net, NetClientDriver::Socket, peer, MODEL, name, cfg)?;
        return Ok(());
    }

    if let Some(host) = &sock.connect {
        // net_socket_connect_init()
        let saddr = parse_host_port(host)?;
        let fd = new_socket(AddressFamily::INET, SocketType::STREAM)
            .map_err(|e| Error::from_io("can't create stream socket", e))?;
        unblock(&fd)?;
        let connected = loop {
            match rustix::net::connect(&fd, &saddr) {
                Ok(()) => break true,
                Err(Errno::INTR | Errno::AGAIN) => continue,
                Err(Errno::INPROGRESS | Errno::ALREADY) => break false,
                Err(e) => return Err(Error::from_io("can't connect socket", e.into())),
            }
        };
        let _ = rustix::net::sockopt::set_tcp_nodelay(&fd, true);
        let mut cfg = SockConfig::new(Framing::Stream, Flavour::Socket);
        cfg.info = format!("socket: connect to {}:{}", saddr.ip(), saddr.port());
        cfg.fd = Some(fd);
        cfg.connecting = !connected;
        new_sock(net, NetClientDriver::Socket, peer, MODEL, name, cfg)?;
        return Ok(());
    }

    if let Some(host) = &sock.mcast {
        // net_socket_mcast_init()
        let saddr = parse_host_port(host)?;
        let local = match &sock.localaddr {
            Some(l) => Some(inet_aton(l).ok_or_else(|| {
                Error::generic(format!("localaddr '{l}' is not a valid IPv4 address"))
            })?),
            None => None,
        };
        let fd = mcast_create(saddr, local)?;
        let mut cfg = SockConfig::new(Framing::Dgram, Flavour::Socket);
        cfg.info = format!("socket: mcast={}:{}", saddr.ip(), saddr.port());
        cfg.fd = Some(fd);
        cfg.dest = Some(SocketAddrAny::from(saddr));
        new_sock(net, NetClientDriver::Socket, peer, MODEL, name, cfg)?;
        return Ok(());
    }

    let Some(udp) = &sock.udp else {
        unreachable!("one of the options is set");
    };
    let Some(localaddr) = &sock.localaddr else {
        return Err(Error::generic("localaddr= is mandatory with udp="));
    };
    // net_socket_udp_init()
    let laddr = parse_host_port(localaddr)?;
    let raddr = parse_host_port(udp)?;
    let fd = udp_bind(laddr)?;
    let mut cfg = SockConfig::new(Framing::Dgram, Flavour::Socket);
    cfg.info = format!("socket: udp={}:{}", raddr.ip(), raddr.port());
    cfg.fd = Some(fd);
    cfg.dest = Some(SocketAddrAny::from(raddr));
    new_sock(net, NetClientDriver::Socket, peer, MODEL, name, cfg)?;
    Ok(())
}
