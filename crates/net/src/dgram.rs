// SPDX-License-Identifier: GPL-2.0-or-later

//! The dgram backend, net/dgram.c: one frame per datagram over UDP, a Unix datagram socket,
//! a multicast group, or a descriptor handed in.

use std::net::{Ipv4Addr, SocketAddrV4};
use std::sync::Arc;

use rustix::net::{AddressFamily, SocketAddrAny, SocketType};
use ruvm_base::{Error, Result};
use ruvm_qapi::types::{NetClientDriver, Netdev, NetdevU, SocketAddress, SocketAddressU};

use crate::client::NetClient;
use crate::net::Net;
use crate::poll::unblock;
use crate::sock::{
    Flavour, Framing, SockConfig, address_type_str, local_address, mcast_create, new_sock,
    new_socket, udp_bind, unix_addr,
};
use crate::util::{convert_host_port, inet_aton};

fn inet_of(addr: &SocketAddress) -> Option<(&str, &str)> {
    match &addr.u {
        SocketAddressU::Inet(i) => Some((&i.host, &i.port)),
        _ => None,
    }
}

fn same_type(a: &SocketAddress, b: &SocketAddress) -> bool {
    std::mem::discriminant(&a.u) == std::mem::discriminant(&b.u)
}

/// `net_dgram_mcast_init()`.
fn mcast_init(
    net: &mut Net,
    peer: Option<&Arc<NetClient>>,
    name: &str,
    saddr: SocketAddrV4,
    local: Option<&SocketAddress>,
) -> Result<()> {
    let (fd, info) = match local.map(|l| &l.u) {
        None => (mcast_create(saddr, None)?, format!("mcast={}:{}", saddr.ip(), saddr.port())),
        Some(SocketAddressU::Inet(l)) => {
            let localaddr = inet_aton(&l.host).ok_or_else(|| {
                Error::generic(format!("localaddr '{}' is not a valid IPv4 address", l.host))
            })?;
            (
                mcast_create(saddr, Some(localaddr))?,
                format!("mcast={}:{}", saddr.ip(), saddr.port()),
            )
        }
        Some(SocketAddressU::Fd(f)) => {
            let raw = net.fd_param(&f.str)?;
            let mut fd = crate::fd::adopt(raw)
                .map_err(|e| Error::from_io("Unable to query local socket address", e))?;
            unblock(&fd)?;
            // The descriptor may be shared with another process, which would then get half
            // of the datagrams. Like QEMU, take the address it is bound to and put a fresh
            // socket joined to that group under the same number.
            let bound = local_address(&fd)?;
            let bound = SocketAddrV4::try_from(bound).unwrap_or(SocketAddrV4::new(0.into(), 0));
            if *bound.ip() == Ipv4Addr::UNSPECIFIED {
                return Err(Error::generic("can't setup multicast destination address"));
            }
            let newfd = mcast_create(bound, None)?;
            rustix::io::dup2(&newfd, &mut fd)
                .map_err(|e| Error::from_io("can't clone the multicast socket", e.into()))?;
            drop(newfd);
            (fd, format!("fd={raw} (cloned mcast={}:{})", saddr.ip(), saddr.port()))
        }
        Some(_) => return Err(Error::generic("only support inet or fd type for local")),
    };
    let mut cfg = SockConfig::new(Framing::Dgram, Flavour::Socket);
    cfg.info = info;
    cfg.fd = Some(fd);
    cfg.dest = Some(SocketAddrAny::from(saddr));
    // QEMU names the model "dram" here, a typo that shows in `info network`.
    new_sock(net, NetClientDriver::Dgram, peer, "dram", name, cfg)?;
    Ok(())
}

/// `net_init_dgram()`.
pub(crate) fn net_init_dgram(
    net: &mut Net,
    netdev: &Netdev,
    name: &str,
    peer: Option<Arc<NetClient>>,
) -> Result<()> {
    let NetdevU::Dgram(opts) = &netdev.u else {
        unreachable!("dgram init with another type");
    };
    let peer = peer.as_ref();
    let remote = opts.remote.as_ref();
    let local = opts.local.as_ref();

    if let Some((host, port)) = remote.and_then(inet_of) {
        let mcast = convert_host_port(host, port)?;
        if mcast.ip().is_multicast() {
            return mcast_init(net, peer, name, mcast, local);
        }
    }

    let Some(local) = local else {
        return Err(Error::generic("dgram requires local= parameter"));
    };
    match remote {
        Some(r) => {
            if matches!(local.u, SocketAddressU::Fd(_)) {
                return Err(Error::generic("don't set remote with local.fd"));
            }
            if !same_type(r, local) {
                return Err(Error::generic("remote and local types must be the same"));
            }
        }
        None => {
            if !matches!(local.u, SocketAddressU::Fd(_)) {
                return Err(Error::generic("type=inet or type=unix requires remote parameter"));
            }
        }
    }

    let mut cfg = SockConfig::new(Framing::Dgram, Flavour::Socket);
    match (&local.u, remote.map(|r| &r.u)) {
        (SocketAddressU::Inet(l), Some(SocketAddressU::Inet(r))) => {
            let laddr = convert_host_port(&l.host, &l.port)?;
            let raddr = convert_host_port(&r.host, &r.port)?;
            cfg.fd = Some(udp_bind(laddr)?);
            cfg.dest = Some(SocketAddrAny::from(raddr));
            cfg.info =
                format!("udp={}:{}/{}:{}", laddr.ip(), laddr.port(), raddr.ip(), raddr.port());
        }
        (SocketAddressU::Unix(l), Some(SocketAddressU::Unix(r))) => {
            if let Err(e) = std::fs::remove_file(&l.path) {
                if e.kind() != std::io::ErrorKind::NotFound {
                    return Err(Error::from_io(format!("failed to unlink socket {}", l.path), e));
                }
            }
            let laddr = unix_addr(&l.path)?;
            let raddr = unix_addr(&r.path)?;
            let fd = new_socket(AddressFamily::UNIX, SocketType::DGRAM)
                .map_err(|e| Error::from_io("can't create datagram socket", e))?;
            rustix::net::bind(&fd, &laddr).map_err(|e| {
                Error::from_io(format!("can't bind unix={} to socket", l.path), e.into())
            })?;
            unblock(&fd)?;
            cfg.fd = Some(fd);
            cfg.dest = Some(SocketAddrAny::from(raddr));
            cfg.info = format!("udp={}:{}", l.path, r.path);
        }
        (SocketAddressU::Fd(f), None) => {
            let raw = net.fd_param(&f.str)?;
            let fd = crate::fd::adopt(raw)
                .map_err(|e| Error::from_io("Unable to query local socket address", e))?;
            unblock(&fd)?;
            cfg.info = match local_address(&fd) {
                Ok(sa) => format!("fd={raw} {}", address_type_str(&sa)),
                Err(_) => format!("fd={raw}"),
            };
            cfg.fd = Some(fd);
        }
        _ => return Err(Error::generic("only support inet or fd type for local")),
    }
    new_sock(net, NetClientDriver::Dgram, peer, "dgram", name, cfg)?;
    Ok(())
}
