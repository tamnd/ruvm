// SPDX-License-Identifier: GPL-2.0-or-later

//! Network backends, ported from QEMU's net/ directory.
//!
//! A [`NetClient`] is one end of a link: a NIC queue on the device side, or a backend such as
//! tap or a socket on the host side. Two clients are peers, and frames sent by one are
//! delivered to the other, or queued while the other cannot take them. Hubs join more than two
//! clients, which is how the old `-net` options wire things up.
//!
//! [`Net`] holds all clients and does what net/net.c does: parsing `-netdev`, `-nic` and
//! `-net`, making the backends, and the monitor side (`info network`, `set_link`,
//! `netdev_del`). Device models attach through [`Net::new_nic`] with their own
//! [`NetClientOps`].
//!
//! Backends here: `tap`, `socket`, `stream`, `dgram`, `hubport`, `user` (libslirp, loaded when
//! the backend is first used), `passt` (Linux only) and `vhost-user`. Each backend with a file
//! descriptor runs a small I/O thread that watches it, since there is no shared event loop yet.

#![deny(unsafe_code)]

mod client;
mod hub;
mod net;
mod opts_visitor;
mod passt;
mod util;

#[cfg(unix)]
mod dgram;
#[cfg(unix)]
mod fd;
#[cfg(unix)]
mod poll;
#[cfg(all(unix, feature = "slirp"))]
mod slirp;
#[cfg(unix)]
mod sock;
#[cfg(unix)]
mod socket;
#[cfg(unix)]
mod stream;
#[cfg(unix)]
mod tap;
#[cfg(target_os = "linux")]
mod tap_linux;
#[cfg(unix)]
mod vhost_user;

pub use client::{
    ETH_ZLEN, MAX_QUEUE_NUM, NET_BUFSIZE, NetClient, NetClientOps, NetOffloads, NetQueue,
    PACKET_FLAG_NONE, PACKET_FLAG_RAW, QUEUE_MAXLEN, SentCb, VNET_HDR_LEN, VNET_HDR_MRG_RXBUF_LEN,
    VNET_HDR_V1_HASH_LEN, VNET_HDR_V1_HASH_TUNNEL_LEN, eth_pad_short_frame,
};
pub use hub::{Hub, hub_id_for_client};
pub use net::{
    AVAILABLE_NETDEVS, DEV_NVECTORS_UNSPECIFIED, EventSink, FdResolver, MAX_NICS, Net, NetEvent,
    Nic, NicConf, NicInfo, netdev_is_modern, parse_modern, show_netdevs,
};
pub use opts_visitor::OptsVisitor;
pub use passt::{PasstOptions, passt_args};
#[cfg(all(unix, feature = "slirp"))]
pub use slirp::libslirp_version;
#[cfg(unix)]
pub use tap::{DEFAULT_NETWORK_DOWN_SCRIPT, DEFAULT_NETWORK_SCRIPT, TapState};
pub use util::{
    MacAddr, SocketReadState, convert_host_port, inet_aton, parse_host_port, parse_macaddr,
};
#[cfg(unix)]
pub use vhost_user::{
    ChardevResolver, VHOST_USER_NET_PROTOCOL_FEATURES, VhostUserChardev, VhostUserNet,
};
