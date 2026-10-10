// SPDX-License-Identifier: GPL-2.0-or-later

//! The stream backend, net/stream.c: frames with a 4 byte length in front over a TCP or
//! Unix stream socket, as a server that takes one client at a time or as a client that may
//! connect again after losing the server.
//!
//! QEMU connects and listens in the background and reports failures only through the info
//! string. Listening happens right away here, but a failure still only shows in the info
//! string, and connecting happens on the backend's I/O thread.

use std::sync::{Arc, Mutex};

use ruvm_base::{Error, Result};
use ruvm_qapi::types::{NetClientDriver, Netdev, NetdevU, SocketAddress, SocketAddressU};

use crate::client::NetClient;
use crate::net::Net;
use crate::poll::unblock;
use crate::sock::{
    Connector, Flavour, Framing, SockConfig, inet_connect, inet_listen, is_abstract, new_sock,
    socket_get_fd, unix_connect, unix_listen,
};

const MODEL: &str = "stream";

/// `socket_listen()`.
fn listen(net: &mut Net, addr: &SocketAddress) -> Result<std::os::fd::OwnedFd> {
    match &addr.u {
        SocketAddressU::Inet(inet) => inet_listen(inet),
        SocketAddressU::Unix(unix) => unix_listen(unix),
        SocketAddressU::Fd(fd) => {
            let sock = socket_get_fd(net, &fd.str)?;
            rustix::net::listen(&sock, 1)
                .map_err(|e| Error::from_io("Failed to listen on fd socket", e.into()))?;
            Ok(sock)
        }
        SocketAddressU::Vsock(_) => Err(Error::generic("socket family AF_VSOCK unsupported")),
    }
}

/// `net_stream_server_init()`.
fn server_init(
    net: &mut Net,
    peer: Option<&Arc<NetClient>>,
    name: &str,
    addr: &SocketAddress,
) -> Result<()> {
    let mut cfg = SockConfig::new(Framing::Stream, Flavour::Stream);
    match listen(net, addr).and_then(|fd| {
        unblock(&fd)?;
        Ok(fd)
    }) {
        Ok(fd) => {
            cfg.info = "listening".to_string();
            cfg.events = net.event_sink.clone();
            cfg.listen_fd = Some(fd);
            if let SocketAddressU::Unix(u) = &addr.u
                && !is_abstract(u)
            {
                cfg.unlink = Some(u.path.clone().into());
            }
        }
        Err(e) => cfg.info = format!("error: {}", e.message()),
    }
    new_sock(net, NetClientDriver::Stream, peer, MODEL, name, cfg)?;
    Ok(())
}

/// `net_stream_client_init()`.
fn client_init(
    net: &mut Net,
    peer: Option<&Arc<NetClient>>,
    name: &str,
    addr: &SocketAddress,
    reconnect_ms: u64,
) -> Result<()> {
    let connector: Connector = match &addr.u {
        SocketAddressU::Inet(inet) => {
            let inet = inet.clone();
            Box::new(move || inet_connect(&inet))
        }
        SocketAddressU::Unix(unix) => {
            let unix = unix.clone();
            Box::new(move || unix_connect(&unix))
        }
        SocketAddressU::Fd(fd) => {
            // The descriptor has to be looked up now, while the monitor is at hand. It can
            // only be used once; connecting again after a hang-up finds it gone, as in QEMU,
            // where the number no longer names a socket by then.
            let fdstr = fd.str.clone();
            let slot = Mutex::new(Some(socket_get_fd(net, &fd.str)?));
            Box::new(move || {
                crate::client::lock(&slot).take().ok_or_else(|| {
                    Error::generic(format!("File descriptor '{fdstr}' is not a socket"))
                })
            })
        }
        SocketAddressU::Vsock(_) => {
            Box::new(|| Err(Error::generic("socket family AF_VSOCK unsupported")))
        }
    };
    let mut cfg = SockConfig::new(Framing::Stream, Flavour::Stream);
    cfg.info = "connecting".to_string();
    cfg.connector = Some(connector);
    cfg.reconnect_ms = reconnect_ms;
    cfg.events = net.event_sink.clone();
    new_sock(net, NetClientDriver::Stream, peer, MODEL, name, cfg)?;
    Ok(())
}

/// `net_init_stream()`.
pub(crate) fn net_init_stream(
    net: &mut Net,
    netdev: &Netdev,
    name: &str,
    peer: Option<Arc<NetClient>>,
) -> Result<()> {
    let NetdevU::Stream(sock) = &netdev.u else {
        unreachable!("stream init with another type");
    };
    if sock.server != Some(true) {
        let ms = sock.reconnect_ms.unwrap_or(0).clamp(0, i64::from(u32::MAX)) as u64;
        return client_init(net, peer.as_ref(), name, &sock.addr, ms);
    }
    if sock.reconnect_ms.is_some() {
        return Err(Error::generic(
            "'reconnect-ms' option is incompatible with socket in server mode",
        ));
    }
    server_init(net, peer.as_ref(), name, &sock.addr)
}
