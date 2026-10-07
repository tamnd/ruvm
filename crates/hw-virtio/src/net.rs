// SPDX-License-Identifier: GPL-2.0-or-later

//! virtio-net, a port of `hw/net/virtio-net.c`.
//!
//! The device has one receive and one transmit queue per queue pair and a control queue at the
//! end: `rx0, tx0, rx1, tx1, ..., ctrl`. Only the first pair exists until the driver negotiates
//! `VIRTIO_NET_F_MQ`, at which point the device grows to every pair it was configured with, and
//! the driver then picks how many it uses through the control queue.
//!
//! Frames go out through a [`NetPeer`], a deliberately small stand in for QEMU's
//! `NetClientState` peer. Frames come in through [`VirtioNet::receive`] and
//! [`VirtioNet::receive_on`], which do what `virtio_net_receive()` does: run the receive filter,
//! then copy the frame and a `virtio_net_hdr` into guest buffers, spreading it over several
//! buffers when mergeable receive buffers were negotiated.
//!
//! Differences from QEMU:
//!
//! - The peer is synchronous. Instead of queueing a frame it cannot take yet, the peer says so
//!   through [`NetPeer::can_receive`] and the device leaves the rest of the transmit queue alone
//!   until [`VirtioNet::tx_resume`] is called. Received frames that find no buffers are not
//!   queued either: [`RxOutcome::NoBuffers`] tells the caller to try again, and
//!   [`NetPeer::rx_ready`] tells the peer when the driver adds buffers.
//! - Transmit runs to completion on every kick. There is no bottom half, no `x-txtimer` and no
//!   `x-txburst`.
//! - When the peer takes virtio-net headers, the transmit path hands it the parsed header rather
//!   than the raw bytes, and a chain shorter than the header is an error in both modes.
//! - Only little endian guests are handled, so the header is never byte swapped.
//! - The link status is set with [`VirtioNet::set_link_status`] and self announcement is started
//!   with [`VirtioNet::announce`], both called by whoever owns the device.
//!
//! Migration state is in the `vmstate` submodule, [`VirtioNetVmState`].
//!
//! Not ported: trace points, QOM registration, vhost and vDPA, RSS and hash reporting
//! (the features stay off and the control commands are refused), receive segment coalescing,
//! UDP tunnel offloads, failover (`VIRTIO_NET_F_STANDBY`), the announce timer and its rounds,
//! the `NIC_RX_FILTER_CHANGED` event, `query-rx-filter`, the dhclient checksum workaround, and
//! the RARP frames QEMU sends for `announce-self` when the guest cannot announce itself.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use ruvm_base::{Error, Result};
use ruvm_virtio_queue::DescriptorChain;

mod vmstate;

pub use vmstate::VirtioNetVmState;

use crate::virtio::{
    VIRTIO_CONFIG_S_DRIVER_OK, VIRTIO_F_VERSION_1, VIRTIO_LEGACY_FEATURES, VIRTIO_QUEUE_MAX,
    VIRTQUEUE_MAX_SIZE, VirtIODevice, VirtioDeviceClass, feature, has_feature,
};

/// `TYPE_VIRTIO_NET`.
pub const TYPE_VIRTIO_NET: &str = "virtio-net-device";

/// `VIRTIO_ID_NET`.
pub const VIRTIO_ID_NET: u16 = 1;

/// `VIRTIO_NET_F_CSUM`: the device handles packets with a partial checksum.
pub const VIRTIO_NET_F_CSUM: u32 = 0;
/// `VIRTIO_NET_F_GUEST_CSUM`: the driver handles packets with a partial checksum.
pub const VIRTIO_NET_F_GUEST_CSUM: u32 = 1;
/// `VIRTIO_NET_F_CTRL_GUEST_OFFLOADS`: the guest offloads can be changed at run time.
pub const VIRTIO_NET_F_CTRL_GUEST_OFFLOADS: u32 = 2;
/// `VIRTIO_NET_F_MTU`: the config space has a valid MTU.
pub const VIRTIO_NET_F_MTU: u32 = 3;
/// `VIRTIO_NET_F_MAC`: the config space has a valid MAC address.
pub const VIRTIO_NET_F_MAC: u32 = 5;
/// `VIRTIO_NET_F_GSO`: legacy, the device handles any GSO type.
pub const VIRTIO_NET_F_GSO: u32 = 6;
/// `VIRTIO_NET_F_GUEST_TSO4`.
pub const VIRTIO_NET_F_GUEST_TSO4: u32 = 7;
/// `VIRTIO_NET_F_GUEST_TSO6`.
pub const VIRTIO_NET_F_GUEST_TSO6: u32 = 8;
/// `VIRTIO_NET_F_GUEST_ECN`.
pub const VIRTIO_NET_F_GUEST_ECN: u32 = 9;
/// `VIRTIO_NET_F_GUEST_UFO`.
pub const VIRTIO_NET_F_GUEST_UFO: u32 = 10;
/// `VIRTIO_NET_F_HOST_TSO4`.
pub const VIRTIO_NET_F_HOST_TSO4: u32 = 11;
/// `VIRTIO_NET_F_HOST_TSO6`.
pub const VIRTIO_NET_F_HOST_TSO6: u32 = 12;
/// `VIRTIO_NET_F_HOST_ECN`.
pub const VIRTIO_NET_F_HOST_ECN: u32 = 13;
/// `VIRTIO_NET_F_HOST_UFO`.
pub const VIRTIO_NET_F_HOST_UFO: u32 = 14;
/// `VIRTIO_NET_F_MRG_RXBUF`: the driver can merge receive buffers.
pub const VIRTIO_NET_F_MRG_RXBUF: u32 = 15;
/// `VIRTIO_NET_F_STATUS`: the config space has a valid status field.
pub const VIRTIO_NET_F_STATUS: u32 = 16;
/// `VIRTIO_NET_F_CTRL_VQ`: there is a control queue.
pub const VIRTIO_NET_F_CTRL_VQ: u32 = 17;
/// `VIRTIO_NET_F_CTRL_RX`: the control queue takes receive mode commands.
pub const VIRTIO_NET_F_CTRL_RX: u32 = 18;
/// `VIRTIO_NET_F_CTRL_VLAN`: the control queue takes VLAN filter commands.
pub const VIRTIO_NET_F_CTRL_VLAN: u32 = 19;
/// `VIRTIO_NET_F_CTRL_RX_EXTRA`: the extra receive mode commands.
pub const VIRTIO_NET_F_CTRL_RX_EXTRA: u32 = 20;
/// `VIRTIO_NET_F_GUEST_ANNOUNCE`: the driver can send gratuitous packets.
pub const VIRTIO_NET_F_GUEST_ANNOUNCE: u32 = 21;
/// `VIRTIO_NET_F_MQ`: several queue pairs.
pub const VIRTIO_NET_F_MQ: u32 = 22;
/// `VIRTIO_NET_F_CTRL_MAC_ADDR`: the MAC address is set through the control queue.
pub const VIRTIO_NET_F_CTRL_MAC_ADDR: u32 = 23;
/// `VIRTIO_NET_F_GUEST_USO4`.
pub const VIRTIO_NET_F_GUEST_USO4: u32 = 54;
/// `VIRTIO_NET_F_GUEST_USO6`.
pub const VIRTIO_NET_F_GUEST_USO6: u32 = 55;
/// `VIRTIO_NET_F_HOST_USO`.
pub const VIRTIO_NET_F_HOST_USO: u32 = 56;
/// `VIRTIO_NET_F_HASH_REPORT`.
pub const VIRTIO_NET_F_HASH_REPORT: u32 = 57;
/// `VIRTIO_NET_F_GUEST_HDRLEN`.
pub const VIRTIO_NET_F_GUEST_HDRLEN: u32 = 59;
/// `VIRTIO_NET_F_RSS`.
pub const VIRTIO_NET_F_RSS: u32 = 60;
/// `VIRTIO_NET_F_RSC_EXT`.
pub const VIRTIO_NET_F_RSC_EXT: u32 = 61;
/// `VIRTIO_NET_F_STANDBY`.
pub const VIRTIO_NET_F_STANDBY: u32 = 62;
/// `VIRTIO_NET_F_SPEED_DUPLEX`: the config space has valid speed and duplex fields.
pub const VIRTIO_NET_F_SPEED_DUPLEX: u32 = 63;

/// The `host_features` virtio-net starts with, the defaults of its feature properties.
/// Multiqueue, RSS, hash reporting and RSC are off.
pub const VIRTIO_NET_DEFAULT_HOST_FEATURES: u64 = feature(VIRTIO_NET_F_CSUM)
    | feature(VIRTIO_NET_F_GUEST_CSUM)
    | feature(VIRTIO_NET_F_GSO)
    | feature(VIRTIO_NET_F_GUEST_TSO4)
    | feature(VIRTIO_NET_F_GUEST_TSO6)
    | feature(VIRTIO_NET_F_GUEST_ECN)
    | feature(VIRTIO_NET_F_GUEST_UFO)
    | feature(VIRTIO_NET_F_GUEST_ANNOUNCE)
    | feature(VIRTIO_NET_F_HOST_TSO4)
    | feature(VIRTIO_NET_F_HOST_TSO6)
    | feature(VIRTIO_NET_F_HOST_ECN)
    | feature(VIRTIO_NET_F_HOST_UFO)
    | feature(VIRTIO_NET_F_MRG_RXBUF)
    | feature(VIRTIO_NET_F_STATUS)
    | feature(VIRTIO_NET_F_CTRL_VQ)
    | feature(VIRTIO_NET_F_CTRL_RX)
    | feature(VIRTIO_NET_F_CTRL_VLAN)
    | feature(VIRTIO_NET_F_CTRL_RX_EXTRA)
    | feature(VIRTIO_NET_F_CTRL_MAC_ADDR)
    | feature(VIRTIO_NET_F_CTRL_GUEST_OFFLOADS)
    | feature(VIRTIO_NET_F_GUEST_USO4)
    | feature(VIRTIO_NET_F_GUEST_USO6)
    | feature(VIRTIO_NET_F_HOST_USO);

/// The guest offloads, the features that say what the driver can receive.
const GUEST_OFFLOADS_MASK: u64 = feature(VIRTIO_NET_F_GUEST_CSUM)
    | feature(VIRTIO_NET_F_GUEST_TSO4)
    | feature(VIRTIO_NET_F_GUEST_TSO6)
    | feature(VIRTIO_NET_F_GUEST_ECN)
    | feature(VIRTIO_NET_F_GUEST_UFO)
    | feature(VIRTIO_NET_F_GUEST_USO4)
    | feature(VIRTIO_NET_F_GUEST_USO6);

/// `VIRTIO_NET_S_LINK_UP`.
pub const VIRTIO_NET_S_LINK_UP: u16 = 1;
/// `VIRTIO_NET_S_ANNOUNCE`: the driver should announce itself.
pub const VIRTIO_NET_S_ANNOUNCE: u16 = 2;

/// `DUPLEX_HALF`.
pub const DUPLEX_HALF: u8 = 0;
/// `DUPLEX_FULL`.
pub const DUPLEX_FULL: u8 = 1;
/// `DUPLEX_UNKNOWN`.
pub const DUPLEX_UNKNOWN: u8 = 0xff;
/// `SPEED_UNKNOWN`.
pub const SPEED_UNKNOWN: i32 = -1;

/// `VIRTIO_NET_RSS_MAX_KEY_SIZE`, reported in config space even without RSS.
pub const VIRTIO_NET_RSS_MAX_KEY_SIZE: u8 = 40;

/// `VIRTIO_NET_HDR_F_NEEDS_CSUM`.
pub const VIRTIO_NET_HDR_F_NEEDS_CSUM: u8 = 1;
/// `VIRTIO_NET_HDR_F_DATA_VALID`.
pub const VIRTIO_NET_HDR_F_DATA_VALID: u8 = 2;
/// `VIRTIO_NET_HDR_F_RSC_INFO`.
pub const VIRTIO_NET_HDR_F_RSC_INFO: u8 = 4;
/// `VIRTIO_NET_HDR_GSO_NONE`.
pub const VIRTIO_NET_HDR_GSO_NONE: u8 = 0;
/// `VIRTIO_NET_HDR_GSO_TCPV4`.
pub const VIRTIO_NET_HDR_GSO_TCPV4: u8 = 1;
/// `VIRTIO_NET_HDR_GSO_UDP`.
pub const VIRTIO_NET_HDR_GSO_UDP: u8 = 3;
/// `VIRTIO_NET_HDR_GSO_TCPV6`.
pub const VIRTIO_NET_HDR_GSO_TCPV6: u8 = 4;
/// `VIRTIO_NET_HDR_GSO_UDP_L4`.
pub const VIRTIO_NET_HDR_GSO_UDP_L4: u8 = 5;
/// `VIRTIO_NET_HDR_GSO_ECN`.
pub const VIRTIO_NET_HDR_GSO_ECN: u8 = 0x80;

/// `sizeof(struct virtio_net_hdr)`, the header without `num_buffers`.
pub const VIRTIO_NET_HDR_LEN: usize = 10;
/// `sizeof(struct virtio_net_hdr_mrg_rxbuf)`, also the size of the virtio 1.0 header.
pub const VIRTIO_NET_HDR_MRG_RXBUF_LEN: usize = 12;

/// `VIRTIO_NET_OK`.
pub const VIRTIO_NET_OK: u8 = 0;
/// `VIRTIO_NET_ERR`.
pub const VIRTIO_NET_ERR: u8 = 1;

/// `VIRTIO_NET_CTRL_RX`.
pub const VIRTIO_NET_CTRL_RX: u8 = 0;
/// `VIRTIO_NET_CTRL_RX_PROMISC`.
pub const VIRTIO_NET_CTRL_RX_PROMISC: u8 = 0;
/// `VIRTIO_NET_CTRL_RX_ALLMULTI`.
pub const VIRTIO_NET_CTRL_RX_ALLMULTI: u8 = 1;
/// `VIRTIO_NET_CTRL_RX_ALLUNI`.
pub const VIRTIO_NET_CTRL_RX_ALLUNI: u8 = 2;
/// `VIRTIO_NET_CTRL_RX_NOMULTI`.
pub const VIRTIO_NET_CTRL_RX_NOMULTI: u8 = 3;
/// `VIRTIO_NET_CTRL_RX_NOUNI`.
pub const VIRTIO_NET_CTRL_RX_NOUNI: u8 = 4;
/// `VIRTIO_NET_CTRL_RX_NOBCAST`.
pub const VIRTIO_NET_CTRL_RX_NOBCAST: u8 = 5;
/// `VIRTIO_NET_CTRL_MAC`.
pub const VIRTIO_NET_CTRL_MAC: u8 = 1;
/// `VIRTIO_NET_CTRL_MAC_TABLE_SET`.
pub const VIRTIO_NET_CTRL_MAC_TABLE_SET: u8 = 0;
/// `VIRTIO_NET_CTRL_MAC_ADDR_SET`.
pub const VIRTIO_NET_CTRL_MAC_ADDR_SET: u8 = 1;
/// `VIRTIO_NET_CTRL_VLAN`.
pub const VIRTIO_NET_CTRL_VLAN: u8 = 2;
/// `VIRTIO_NET_CTRL_VLAN_ADD`.
pub const VIRTIO_NET_CTRL_VLAN_ADD: u8 = 0;
/// `VIRTIO_NET_CTRL_VLAN_DEL`.
pub const VIRTIO_NET_CTRL_VLAN_DEL: u8 = 1;
/// `VIRTIO_NET_CTRL_ANNOUNCE`.
pub const VIRTIO_NET_CTRL_ANNOUNCE: u8 = 3;
/// `VIRTIO_NET_CTRL_ANNOUNCE_ACK`.
pub const VIRTIO_NET_CTRL_ANNOUNCE_ACK: u8 = 0;
/// `VIRTIO_NET_CTRL_MQ`.
pub const VIRTIO_NET_CTRL_MQ: u8 = 4;
/// `VIRTIO_NET_CTRL_MQ_VQ_PAIRS_SET`.
pub const VIRTIO_NET_CTRL_MQ_VQ_PAIRS_SET: u8 = 0;
/// `VIRTIO_NET_CTRL_MQ_RSS_CONFIG`.
pub const VIRTIO_NET_CTRL_MQ_RSS_CONFIG: u8 = 1;
/// `VIRTIO_NET_CTRL_MQ_HASH_CONFIG`.
pub const VIRTIO_NET_CTRL_MQ_HASH_CONFIG: u8 = 2;
/// `VIRTIO_NET_CTRL_MQ_VQ_PAIRS_MIN`.
pub const VIRTIO_NET_CTRL_MQ_VQ_PAIRS_MIN: u16 = 1;
/// `VIRTIO_NET_CTRL_MQ_VQ_PAIRS_MAX`.
pub const VIRTIO_NET_CTRL_MQ_VQ_PAIRS_MAX: u16 = 0x8000;
/// `VIRTIO_NET_CTRL_GUEST_OFFLOADS`.
pub const VIRTIO_NET_CTRL_GUEST_OFFLOADS: u8 = 5;
/// `VIRTIO_NET_CTRL_GUEST_OFFLOADS_SET`.
pub const VIRTIO_NET_CTRL_GUEST_OFFLOADS_SET: u8 = 0;

/// `MAC_TABLE_ENTRIES`: how many addresses the receive filter table holds.
pub const MAC_TABLE_ENTRIES: usize = 64;
/// `MAX_VLAN`: VLAN ids go from 0 to 4095.
pub const MAX_VLAN: u16 = 4096;
/// `NET_BUFSIZE`: the largest frame the net layer passes on. Longer ones are dropped.
pub const NET_BUFSIZE: usize = 4096 + 65536;

/// `VIRTIO_NET_RX_QUEUE_DEFAULT_SIZE`.
pub const VIRTIO_NET_RX_QUEUE_DEFAULT_SIZE: u16 = 256;
/// `VIRTIO_NET_TX_QUEUE_DEFAULT_SIZE`.
pub const VIRTIO_NET_TX_QUEUE_DEFAULT_SIZE: u16 = 256;
/// `VIRTIO_NET_RX_QUEUE_MIN_SIZE`.
pub const VIRTIO_NET_RX_QUEUE_MIN_SIZE: u16 = 256;
/// `VIRTIO_NET_TX_QUEUE_MIN_SIZE`.
pub const VIRTIO_NET_TX_QUEUE_MIN_SIZE: u16 = 256;
/// `virtio_net_max_tx_queue_size()` without a vhost-user peer.
pub const VIRTIO_NET_TX_QUEUE_MAX_SIZE: u16 = 256;
/// The size of the control queue.
pub const VIRTIO_NET_CTRL_QUEUE_SIZE: u16 = 64;

/// The config space layout, `struct virtio_net_config`.
const CFG_MAC: usize = 0;
const CFG_STATUS: usize = 6;
const CFG_MAX_VQ_PAIRS: usize = 8;
const CFG_MTU: usize = 10;
const CFG_SPEED: usize = 12;
const CFG_DUPLEX: usize = 16;
const CFG_RSS_MAX_KEY_SIZE: usize = 17;
const CFG_RSS_MAX_INDIRECTION_TABLE_LENGTH: usize = 18;
const CFG_SUPPORTED_HASH_TYPES: usize = 20;
/// `sizeof(struct virtio_net_config)`.
pub const VIRTIO_NET_CONFIG_SIZE: usize = 24;

/// `virtio_net_hdr_mrg_rxbuf`: the header in front of every frame.
///
/// Legacy drivers without mergeable receive buffers use only the first
/// [`VIRTIO_NET_HDR_LEN`] bytes. Everything else uses all [`VIRTIO_NET_HDR_MRG_RXBUF_LEN`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VirtioNetHdr {
    /// `VIRTIO_NET_HDR_F_*`.
    pub flags: u8,
    /// `VIRTIO_NET_HDR_GSO_*`.
    pub gso_type: u8,
    /// The length of the headers the device needs to replicate for each segment.
    pub hdr_len: u16,
    /// The size of each segment.
    pub gso_size: u16,
    /// Where checksumming starts.
    pub csum_start: u16,
    /// Where the checksum goes, counted from `csum_start`.
    pub csum_offset: u16,
    /// How many buffers a merged frame spans.
    pub num_buffers: u16,
}

impl VirtioNetHdr {
    /// The 12 byte little endian form.
    pub fn to_bytes(&self) -> [u8; VIRTIO_NET_HDR_MRG_RXBUF_LEN] {
        let mut b = [0; VIRTIO_NET_HDR_MRG_RXBUF_LEN];
        b[0] = self.flags;
        b[1] = self.gso_type;
        b[2..4].copy_from_slice(&self.hdr_len.to_le_bytes());
        b[4..6].copy_from_slice(&self.gso_size.to_le_bytes());
        b[6..8].copy_from_slice(&self.csum_start.to_le_bytes());
        b[8..10].copy_from_slice(&self.csum_offset.to_le_bytes());
        b[10..12].copy_from_slice(&self.num_buffers.to_le_bytes());
        b
    }

    /// Parses a header. `num_buffers` is only read when `b` is at least 12 bytes long. Returns
    /// `None` if `b` is shorter than [`VIRTIO_NET_HDR_LEN`].
    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        if b.len() < VIRTIO_NET_HDR_LEN {
            return None;
        }
        let word = |i: usize| u16::from_le_bytes([b[i], b[i + 1]]);
        Some(VirtioNetHdr {
            flags: b[0],
            gso_type: b[1],
            hdr_len: word(2),
            gso_size: word(4),
            csum_start: word(6),
            csum_offset: word(8),
            num_buffers: if b.len() >= VIRTIO_NET_HDR_MRG_RXBUF_LEN { word(10) } else { 0 },
        })
    }
}

/// The other end of the NIC, a small stand in for the `NetClientState` peer.
///
/// Every method takes the queue pair it is about, since each pair has its own peer queue in
/// QEMU. Only [`send`](Self::send) has to be written.
pub trait NetPeer: Send + fmt::Debug {
    /// `peer_has_vnet_hdr()`: the peer takes and gives frames with a virtio-net header, so
    /// offloads can be passed through.
    fn has_vnet_hdr(&self) -> bool {
        false
    }

    /// `peer_has_ufo()`.
    fn has_ufo(&self) -> bool {
        false
    }

    /// `peer_has_uso()`.
    fn has_uso(&self) -> bool {
        false
    }

    /// Whether the peer can take a frame from queue pair `pair` now. When it cannot, the rest of
    /// the transmit queue waits for [`VirtioNet::tx_resume`].
    fn can_receive(&self, _pair: u16) -> bool {
        true
    }

    /// A frame the guest sent on queue pair `pair`. `hdr` is the guest's header when the peer
    /// takes virtio-net headers, and `None` otherwise.
    fn send(&mut self, pair: u16, hdr: Option<&VirtioNetHdr>, frame: &[u8]);

    /// `qemu_set_offload()`: the guest offloads, a mask of `VIRTIO_NET_F_GUEST_*` bits, changed.
    fn set_offloads(&mut self, _offloads: u64) {}

    /// `qemu_flush_queued_packets()`: the driver made receive buffers available on queue pair
    /// `pair`, so frames that got [`RxOutcome::NoBuffers`] can be tried again.
    fn rx_ready(&mut self, _pair: u16) {}
}

/// The virtio-net properties, `NICConf` and `virtio_net_conf`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VirtioNetConf {
    /// `mac`.
    pub mac: [u8; 6],
    /// The feature properties (`csum`, `mrg_rxbuf`, `mq` and so on) as one mask.
    pub host_features: u64,
    /// `rx_queue_size`.
    pub rx_queue_size: u16,
    /// `tx_queue_size`.
    pub tx_queue_size: u16,
    /// `host_mtu`, 0 for none.
    pub host_mtu: u16,
    /// `speed` in Mbit/s, or [`SPEED_UNKNOWN`].
    pub speed: i32,
    /// `duplex`: `"half"`, `"full"` or unset.
    pub duplex: Option<String>,
    /// How many queue pairs the peer has, `peers.queues`.
    pub queue_pairs: u32,
}

impl Default for VirtioNetConf {
    fn default() -> Self {
        VirtioNetConf {
            // qemu_macaddr_default_if_unset() hands out 52:54:00:12:34:56 first.
            mac: [0x52, 0x54, 0x00, 0x12, 0x34, 0x56],
            host_features: VIRTIO_NET_DEFAULT_HOST_FEATURES,
            rx_queue_size: VIRTIO_NET_RX_QUEUE_DEFAULT_SIZE,
            tx_queue_size: VIRTIO_NET_TX_QUEUE_DEFAULT_SIZE,
            host_mtu: 0,
            speed: SPEED_UNKNOWN,
            duplex: None,
            queue_pairs: 1,
        }
    }
}

/// The receive filter address table, `n->mac_table`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MacTable {
    /// The unicast addresses followed by the multicast ones.
    pub macs: Vec<[u8; 6]>,
    /// Where the multicast addresses start.
    pub first_multi: usize,
    /// The driver sent more unicast addresses than fit, so all unicast frames pass.
    pub uni_overflow: bool,
    /// The driver sent more multicast addresses than fit, so all multicast frames pass.
    pub multi_overflow: bool,
}

/// What became of a frame handed to [`VirtioNet::receive`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RxOutcome {
    /// The frame is in guest buffers.
    Delivered,
    /// The receive filter dropped it. It counts as consumed.
    Filtered,
    /// It was dropped because it does not fit, or the link is down. It counts as consumed.
    Dropped,
    /// There are not enough guest buffers right now. Try again after [`NetPeer::rx_ready`].
    NoBuffers,
    /// The queue pair cannot receive: it is not in use, not set up, or the driver is not ready.
    NotReady,
    /// The driver did something wrong and the device is now broken.
    Failed,
}

/// The per queue pair state, `VirtIONetQueue`.
#[derive(Clone, Copy, Debug, Default)]
struct NetQueue {
    /// The peer could not take a frame and the transmit queue is waiting for it.
    tx_waiting: bool,
}

/// The virtio-net device model, `VirtIONet`.
#[derive(Debug)]
pub struct VirtioNet {
    conf: VirtioNetConf,
    peer: Option<Box<dyn NetPeer>>,
    host_features: u64,
    config_size: usize,
    duplex: u8,
    mac: [u8; 6],
    status: u16,
    max_queue_pairs: u16,
    curr_queue_pairs: u16,
    multiqueue: bool,
    vqs: Vec<NetQueue>,
    has_vnet_hdr: bool,
    mergeable_rx_bufs: bool,
    guest_hdr_len: usize,
    host_hdr_len: usize,
    curr_guest_offloads: u64,
    promisc: bool,
    allmulti: bool,
    alluni: bool,
    nomulti: bool,
    nouni: bool,
    nobcast: bool,
    mac_table: MacTable,
    vlans: Vec<u32>,
    /// `saved_guest_offloads`: the migrated offloads, between loading them and `post_load`.
    saved_guest_offloads: Option<u64>,
}

impl VirtioNet {
    /// A device with properties `conf` whose frames go to `peer`. Without a peer, transmitted
    /// frames are dropped. The properties are checked when the device is realized.
    pub fn new(conf: VirtioNetConf, peer: Option<Box<dyn NetPeer>>) -> Self {
        let mac = conf.mac;
        VirtioNet {
            conf,
            peer,
            host_features: 0,
            config_size: 0,
            duplex: DUPLEX_UNKNOWN,
            mac,
            status: 0,
            max_queue_pairs: 1,
            curr_queue_pairs: 1,
            multiqueue: false,
            vqs: Vec::new(),
            has_vnet_hdr: false,
            mergeable_rx_bufs: false,
            guest_hdr_len: VIRTIO_NET_HDR_LEN,
            host_hdr_len: 0,
            curr_guest_offloads: 0,
            promisc: true,
            allmulti: false,
            alluni: false,
            nomulti: false,
            nouni: false,
            nobcast: false,
            mac_table: MacTable::default(),
            vlans: vec![u32::MAX; usize::from(MAX_VLAN >> 5)],
            saved_guest_offloads: None,
        }
    }

    /// The properties.
    pub fn conf(&self) -> &VirtioNetConf {
        &self.conf
    }

    /// The peer, if there is one.
    pub fn peer(&self) -> Option<&dyn NetPeer> {
        self.peer.as_deref()
    }

    /// The peer, mutably.
    pub fn peer_mut(&mut self) -> Option<&mut (dyn NetPeer + 'static)> {
        self.peer.as_deref_mut()
    }

    /// The current MAC address, which the driver may have changed.
    pub fn mac(&self) -> [u8; 6] {
        self.mac
    }

    /// The config space status field, `VIRTIO_NET_S_*`.
    pub fn status(&self) -> u16 {
        self.status
    }

    /// Whether the link is up.
    pub fn link_up(&self) -> bool {
        self.status & VIRTIO_NET_S_LINK_UP != 0
    }

    /// How many queue pairs the device has at most.
    pub fn max_queue_pairs(&self) -> u16 {
        self.max_queue_pairs
    }

    /// How many queue pairs the driver uses.
    pub fn curr_queue_pairs(&self) -> u16 {
        self.curr_queue_pairs
    }

    /// Whether the driver negotiated multiqueue, so every pair exists.
    pub fn multiqueue(&self) -> bool {
        self.multiqueue
    }

    /// Whether the driver negotiated mergeable receive buffers.
    pub fn mergeable_rx_bufs(&self) -> bool {
        self.mergeable_rx_bufs
    }

    /// The size of the header in front of every frame in guest buffers.
    pub fn guest_hdr_len(&self) -> usize {
        self.guest_hdr_len
    }

    /// The size of the header the peer sees, 0 when it takes none.
    pub fn host_hdr_len(&self) -> usize {
        self.host_hdr_len
    }

    /// The guest offloads in effect, a mask of `VIRTIO_NET_F_GUEST_*` bits.
    pub fn curr_guest_offloads(&self) -> u64 {
        self.curr_guest_offloads
    }

    /// Receive everything.
    pub fn promisc(&self) -> bool {
        self.promisc
    }

    /// Receive every multicast frame.
    pub fn allmulti(&self) -> bool {
        self.allmulti
    }

    /// Receive every unicast frame.
    pub fn alluni(&self) -> bool {
        self.alluni
    }

    /// Receive no multicast frames.
    pub fn nomulti(&self) -> bool {
        self.nomulti
    }

    /// Receive no unicast frames.
    pub fn nouni(&self) -> bool {
        self.nouni
    }

    /// Receive no broadcast frames.
    pub fn nobcast(&self) -> bool {
        self.nobcast
    }

    /// The receive filter address table.
    pub fn mac_table(&self) -> &MacTable {
        &self.mac_table
    }

    /// Whether frames tagged with VLAN `vid` get through the filter.
    pub fn vlan_allowed(&self, vid: u16) -> bool {
        vid < MAX_VLAN && self.vlans[usize::from(vid >> 5)] & (1 << (vid & 0x1f)) != 0
    }

    /// The index of the receive queue of pair `pair`.
    pub const fn rx_queue(pair: u16) -> u16 {
        pair * 2
    }

    /// The index of the transmit queue of pair `pair`.
    pub const fn tx_queue(pair: u16) -> u16 {
        pair * 2 + 1
    }

    /// The index of the control queue, which is always the last one.
    pub fn ctrl_queue(vdev: &VirtIODevice) -> u16 {
        vdev.num_queues().saturating_sub(1) as u16
    }

    fn config_size(host_features: u64) -> usize {
        // virtio_net_set_config_size(), which counts VIRTIO_NET_F_MAC as always on.
        let f = host_features | feature(VIRTIO_NET_F_MAC);
        let mut size = CFG_MAC + 6;
        let sizes = [
            (feature(VIRTIO_NET_F_STATUS), CFG_STATUS + 2),
            (feature(VIRTIO_NET_F_MQ), CFG_MAX_VQ_PAIRS + 2),
            (feature(VIRTIO_NET_F_MTU), CFG_MTU + 2),
            (feature(VIRTIO_NET_F_SPEED_DUPLEX), CFG_DUPLEX + 1),
            (feature(VIRTIO_NET_F_RSS) | feature(VIRTIO_NET_F_HASH_REPORT), VIRTIO_NET_CONFIG_SIZE),
        ];
        for (mask, end) in sizes {
            if f & mask != 0 {
                size = size.max(end);
            }
        }
        size
    }

    /// `virtio_net_started()`.
    fn started(&self, status: u8) -> bool {
        status & VIRTIO_CONFIG_S_DRIVER_OK != 0 && self.link_up()
    }

    /// `virtio_net_change_num_queues()` and `virtio_net_set_queue_pairs()`: grows or shrinks
    /// the device to `max` pairs. The pairs that stay keep their state, and the control queue
    /// is always added back last.
    fn set_multiqueue(&mut self, vdev: &mut VirtIODevice, multiqueue: bool) {
        let max = if multiqueue { self.max_queue_pairs } else { 1 };
        self.multiqueue = multiqueue;
        let new_num = usize::from(max) * 2 + 1;
        let old_num = vdev.num_queues();
        if old_num == new_num {
            return;
        }
        // Drop the control queue, then the pairs that go away, then add the new pairs and the
        // control queue again.
        let pairs_kept = (old_num - 1).min(new_num - 1);
        vdev.truncate_queues(pairs_kept);
        let mut added = Ok(());
        for _ in (pairs_kept / 2)..usize::from(max) {
            added = added
                .and_then(|_| vdev.add_queue(self.conf.rx_queue_size))
                .and_then(|_| vdev.add_queue(self.conf.tx_queue_size))
                .map(|_| ());
        }
        added = added.and_then(|_| vdev.add_queue(VIRTIO_NET_CTRL_QUEUE_SIZE).map(|_| ()));
        if let Err(e) = added {
            // realize made sure the queues fit, so this cannot happen.
            vdev.error(&e.to_string());
        }
    }

    /// `virtio_net_set_mrg_rx_bufs()`.
    fn set_mrg_rx_bufs(&mut self, mergeable: bool, version_1: bool) {
        self.mergeable_rx_bufs = mergeable;
        self.guest_hdr_len =
            if version_1 || mergeable { VIRTIO_NET_HDR_MRG_RXBUF_LEN } else { VIRTIO_NET_HDR_LEN };
        if self.has_vnet_hdr {
            self.host_hdr_len = self.guest_hdr_len;
        }
    }

    /// `virtio_net_apply_guest_offloads()`.
    fn apply_guest_offloads(&mut self) {
        let offloads = self.curr_guest_offloads;
        if let Some(peer) = self.peer.as_mut() {
            peer.set_offloads(offloads);
        }
    }

    // Receive.

    /// `virtio_net_can_receive()` for queue pair `pair`.
    pub fn can_receive(&self, vdev: &VirtIODevice, pair: u16) -> bool {
        pair < self.curr_queue_pairs
            && vdev.queue_ready(Self::rx_queue(pair))
            && vdev.status() & VIRTIO_CONFIG_S_DRIVER_OK != 0
    }

    /// `virtio_net_has_buffers()`: whether the driver made room for `bufsize` bytes. Turns
    /// queue notifications on when it did not, so the driver kicks once it adds buffers, and
    /// off when it did.
    fn has_buffers(&self, vdev: &mut VirtIODevice, rxq: u16, bufsize: u64) -> bool {
        if vdev.queue_empty(rxq) || self.mergeable_rx_bufs {
            let (in_bytes, _) = vdev.avail_bytes(rxq, bufsize, 0);
            if in_bytes < bufsize {
                vdev.set_queue_notification(rxq, true);
                return false;
            }
        }
        vdev.set_queue_notification(rxq, false);
        true
    }

    /// `receive_filter()`: whether `frame` gets past the receive filter.
    pub fn receive_filter(&self, frame: &[u8]) -> bool {
        const BCAST: [u8; 6] = [0xff; 6];
        if self.promisc {
            return true;
        }
        if frame.len() < 14 {
            // A truncated Ethernet frame.
            return false;
        }
        if frame[12..14] == [0x81, 0x00] {
            if frame.len() < 16 {
                // A truncated VLAN frame.
                return false;
            }
            let vid = u16::from_be_bytes([frame[14], frame[15]]) & 0xfff;
            if !self.vlan_allowed(vid) {
                return false;
            }
        }
        let dst = &frame[..6];
        let table = &self.mac_table;
        if dst[0] & 1 != 0 {
            if dst == BCAST {
                return !self.nobcast;
            }
            if self.nomulti {
                return false;
            }
            if self.allmulti || table.multi_overflow {
                return true;
            }
            table.macs[table.first_multi..].iter().any(|m| m == dst)
        } else {
            if self.nouni {
                return false;
            }
            if self.alluni || table.uni_overflow || dst == self.mac {
                return true;
            }
            table.macs[..table.first_multi].iter().any(|m| m == dst)
        }
    }

    /// Receives `frame` on the first queue pair, with no offload header.
    pub fn receive(&mut self, vdev: &mut VirtIODevice, frame: &[u8]) -> RxOutcome {
        self.receive_on(vdev, 0, None, frame)
    }

    /// `virtio_net_receive()`: receives `frame` on queue pair `pair`.
    ///
    /// `hdr` is the offload header from a peer that uses virtio-net headers. It is written to
    /// the guest as is, except for `num_buffers`. Without one, or when the peer does not use
    /// headers, the guest gets a header that says there is nothing to offload.
    pub fn receive_on(
        &mut self,
        vdev: &mut VirtIODevice,
        pair: u16,
        hdr: Option<&VirtioNetHdr>,
        frame: &[u8],
    ) -> RxOutcome {
        if !self.link_up() {
            // qemu_deliver_packet_iov() drops what arrives for a NIC whose link is down.
            return RxOutcome::Dropped;
        }
        if !self.can_receive(vdev, pair) {
            return RxOutcome::NotReady;
        }
        let rxq = Self::rx_queue(pair);
        let guest_hdr_len = self.guest_hdr_len;
        if !self.has_buffers(vdev, rxq, (frame.len() + guest_hdr_len) as u64) {
            return RxOutcome::NoBuffers;
        }
        if !self.receive_filter(frame) {
            return RxOutcome::Filtered;
        }

        let header = match hdr {
            Some(h) if self.has_vnet_hdr => *h,
            _ => VirtioNetHdr { gso_type: VIRTIO_NET_HDR_GSO_NONE, ..VirtioNetHdr::default() },
        };
        let header = header.to_bytes();
        let mem = Arc::clone(vdev.mem());
        // QEMU pretends the frame starts with the host header, which the loop below always
        // steps over first. That is why a frame with an empty payload still takes a buffer
        // when the peer uses headers.
        let size = self.host_hdr_len + frame.len();
        let mut offset = 0;
        let mut elems: Vec<(DescriptorChain, u32)> = Vec::new();

        while offset < size {
            if elems.len() == usize::from(VIRTQUEUE_MAX_SIZE) {
                vdev.error("virtio-net unexpected long buffer chain");
                return self.rx_fail(vdev, rxq, elems, RxOutcome::Failed);
            }
            if !self.mergeable_rx_bufs {
                // QEMU copies what fits and then gives the buffer back with virtqueue_unpop()
                // if the frame did not fit. Looking first does the same without the unpop.
                if let Some(chain) = vdev.peek(rxq) {
                    let room = chain.writable_len().saturating_sub(guest_hdr_len as u64);
                    if !chain.writable().is_empty() && room < frame.len() as u64 {
                        return RxOutcome::Dropped;
                    }
                }
            }
            let Some(chain) = vdev.pop(rxq) else {
                if !elems.is_empty() {
                    vdev.error(&format!(
                        "virtio-net unexpected empty queue: i {} mergeable {} offset {offset}, \
                         size {size}, guest hdr len {guest_hdr_len}, host hdr len {} guest \
                         features {:#x}",
                        elems.len(),
                        u8::from(self.mergeable_rx_bufs),
                        self.host_hdr_len,
                        vdev.guest_features()
                    ));
                }
                return self.rx_fail(vdev, rxq, elems, RxOutcome::Failed);
            };
            if chain.writable().is_empty() {
                vdev.error("virtio-net receive queue contains no in buffers");
                vdev.detach(rxq, &chain);
                return self.rx_fail(vdev, rxq, elems, RxOutcome::Failed);
            }
            let mut w = chain.writer(&*mem);
            let mut total = 0;
            if elems.is_empty() {
                // receive_header(): only the fields before num_buffers.
                let n = w.write(&header[..VIRTIO_NET_HDR_LEN]).unwrap_or(0);
                w.skip((guest_hdr_len - n) as u64);
                offset = self.host_hdr_len;
                total += guest_hdr_len;
            }
            let data = &frame[offset - self.host_hdr_len..];
            let len = w.write(data).unwrap_or(0);
            total += len;
            offset += len;
            elems.push((chain, total as u32));
        }

        if self.mergeable_rx_bufs {
            if let Some((first, _)) = elems.first() {
                let mut w = first.writer(&*mem);
                if w.skip(VIRTIO_NET_HDR_LEN as u64) == VIRTIO_NET_HDR_LEN as u64 {
                    let _ = w.write(&(elems.len() as u16).to_le_bytes());
                }
            }
        }
        for (chain, len) in &elems {
            vdev.push(rxq, chain, *len);
        }
        vdev.notify(rxq);
        RxOutcome::Delivered
    }

    /// The error exit of `virtio_net_receive_rcu()`: forgets the buffers taken so far.
    fn rx_fail(
        &self,
        vdev: &mut VirtIODevice,
        rxq: u16,
        elems: Vec<(DescriptorChain, u32)>,
        outcome: RxOutcome,
    ) -> RxOutcome {
        for (chain, _) in &elems {
            vdev.detach(rxq, chain);
        }
        outcome
    }

    // Transmit.

    /// `virtqueue_drop_all()` on the transmit queue of `pair`.
    fn drop_tx_queue_data(vdev: &mut VirtIODevice, txq: u16) {
        let mut dropped = 0;
        while let Some(chain) = vdev.pop(txq) {
            vdev.push(txq, &chain, 0);
            dropped += 1;
        }
        if dropped > 0 {
            vdev.notify(txq);
        }
    }

    /// `virtio_net_flush_tx()`: sends everything on the transmit queue of `pair`. Returns how
    /// many frames were sent or dropped.
    fn flush_tx(&mut self, vdev: &mut VirtIODevice, pair: u16) -> usize {
        let txq = Self::tx_queue(pair);
        if vdev.status() & VIRTIO_CONFIG_S_DRIVER_OK == 0 {
            return 0;
        }
        let mem = Arc::clone(vdev.mem());
        let mut sent = 0;
        loop {
            if self.peer.as_ref().is_some_and(|p| !p.can_receive(pair)) {
                // QEMU queues the frame in the net layer and waits for the peer to call back.
                // Here the chains stay where they are until tx_resume().
                if !vdev.queue_empty(txq) {
                    vdev.set_queue_notification(txq, false);
                    self.vqs[usize::from(pair)].tx_waiting = true;
                }
                return sent;
            }
            let Some(chain) = vdev.pop(txq) else {
                break;
            };
            if chain.readable().is_empty() {
                vdev.error("virtio-net header not in first element");
                vdev.detach(txq, &chain);
                break;
            }
            let total = chain.readable_len();
            if total < self.guest_hdr_len as u64 {
                vdev.error("virtio-net header is invalid");
                vdev.detach(txq, &chain);
                break;
            }
            let frame_len = total - self.guest_hdr_len as u64;
            if frame_len == 0 && self.host_hdr_len == 0 {
                vdev.error("virtio-net nothing to send");
                vdev.detach(txq, &chain);
                break;
            }
            // qemu_sendv_packet_async() drops what is too big without reading it, which is
            // what makes the large_tx qtest pass with descriptors far bigger than RAM.
            if frame_len + self.host_hdr_len as u64 <= NET_BUFSIZE as u64 && self.link_up() {
                let mut r = chain.reader(&*mem);
                let mut hdr_bytes = vec![0; self.guest_hdr_len];
                let read = r.read_exact(&mut hdr_bytes).and_then(|()| r.read_to_vec());
                match read {
                    Ok(frame) => {
                        let hdr = if self.has_vnet_hdr {
                            VirtioNetHdr::from_bytes(&hdr_bytes)
                        } else {
                            None
                        };
                        if let Some(peer) = self.peer.as_mut() {
                            peer.send(pair, hdr.as_ref(), &frame);
                        }
                    }
                    Err(e) => {
                        vdev.error(&format!("virtio-net: {e}"));
                        vdev.detach(txq, &chain);
                        break;
                    }
                }
            }
            vdev.push(txq, &chain, 0);
            vdev.notify(txq);
            sent += 1;
        }
        sent
    }

    /// `virtio_net_handle_tx_bh()`: the driver kicked the transmit queue of `pair`.
    fn handle_tx(&mut self, vdev: &mut VirtIODevice, pair: u16) {
        let txq = Self::tx_queue(pair);
        if !self.link_up() {
            Self::drop_tx_queue_data(vdev, txq);
            return;
        }
        if usize::from(pair) >= self.vqs.len() || self.vqs[usize::from(pair)].tx_waiting {
            return;
        }
        self.flush_tx(vdev, pair);
    }

    /// `virtio_net_tx_complete()`: the peer can take frames on `pair` again. Sends whatever
    /// the transmit queue still holds.
    pub fn tx_resume(&mut self, vdev: &mut VirtIODevice, pair: u16) {
        let Some(q) = self.vqs.get_mut(usize::from(pair)) else {
            return;
        };
        if !q.tx_waiting {
            return;
        }
        q.tx_waiting = false;
        vdev.set_queue_notification(Self::tx_queue(pair), true);
        if self.link_up() {
            self.flush_tx(vdev, pair);
        } else {
            Self::drop_tx_queue_data(vdev, Self::tx_queue(pair));
        }
    }

    // Link status and announce.

    /// `virtio_net_set_link_status()`: the link went up or down. The driver gets a config
    /// interrupt if that changes the status.
    pub fn set_link_status(&mut self, vdev: &mut VirtIODevice, up: bool) {
        let old = self.status;
        if up {
            self.status |= VIRTIO_NET_S_LINK_UP;
        } else {
            self.status &= !VIRTIO_NET_S_LINK_UP;
        }
        if self.status != old {
            vdev.notify_config();
        }
        self.update_queues(vdev, vdev.status());
    }

    /// `virtio_net_announce()`: asks the driver to announce itself, if it negotiated
    /// `VIRTIO_NET_F_GUEST_ANNOUNCE` and has a control queue to acknowledge it on. Returns
    /// whether it was asked.
    pub fn announce(&mut self, vdev: &mut VirtIODevice) -> bool {
        if vdev.has_feature(VIRTIO_NET_F_GUEST_ANNOUNCE) && vdev.has_feature(VIRTIO_NET_F_CTRL_VQ) {
            self.status |= VIRTIO_NET_S_ANNOUNCE;
            vdev.notify_config();
            true
        } else {
            false
        }
    }

    /// The queue part of `virtio_net_set_status()`.
    fn update_queues(&mut self, vdev: &mut VirtIODevice, status: u8) {
        for pair in 0..self.max_queue_pairs {
            let queue_status = if (!self.multiqueue && pair != 0) || pair >= self.curr_queue_pairs {
                0
            } else {
                status
            };
            let started = self.started(queue_status);
            if started {
                if let Some(peer) = self.peer.as_mut() {
                    peer.rx_ready(pair);
                }
            }
            if !self.vqs[usize::from(pair)].tx_waiting {
                continue;
            }
            if !started && !self.link_up() && queue_status & VIRTIO_CONFIG_S_DRIVER_OK != 0 {
                self.vqs[usize::from(pair)].tx_waiting = false;
                vdev.set_queue_notification(Self::tx_queue(pair), true);
                Self::drop_tx_queue_data(vdev, Self::tx_queue(pair));
            }
        }
    }

    // The control queue.

    /// `virtio_net_handle_ctrl()`.
    fn handle_ctrl(&mut self, vdev: &mut VirtIODevice, ctrlq: u16) {
        let mem = Arc::clone(vdev.mem());
        while let Some(chain) = vdev.pop(ctrlq) {
            if chain.writable_len() < 1 || chain.readable_len() < 2 {
                vdev.error("virtio-net ctrl missing headers");
                vdev.detach(ctrlq, &chain);
                break;
            }
            let out = chain.reader(&*mem).read_to_vec();
            let status = match out {
                Ok(out) => self.handle_ctrl_cmd(vdev, out[0], out[1], &out[2..]),
                Err(e) => {
                    vdev.error(&format!("virtio-net: {e}"));
                    vdev.detach(ctrlq, &chain);
                    break;
                }
            };
            let _ = chain.writer(&*mem).write(&[status]);
            vdev.push(ctrlq, &chain, 1);
            vdev.notify(ctrlq);
        }
    }

    /// `virtio_net_handle_ctrl_iov()` once the header is split off: runs one command and
    /// returns the ack.
    fn handle_ctrl_cmd(&mut self, vdev: &mut VirtIODevice, class: u8, cmd: u8, data: &[u8]) -> u8 {
        let ok = match class {
            VIRTIO_NET_CTRL_RX => self.handle_rx_mode(cmd, data),
            VIRTIO_NET_CTRL_MAC => self.handle_mac(cmd, data),
            VIRTIO_NET_CTRL_VLAN => self.handle_vlan_table(cmd, data),
            VIRTIO_NET_CTRL_ANNOUNCE => self.handle_announce(cmd),
            VIRTIO_NET_CTRL_MQ => self.handle_mq(vdev, cmd, data),
            VIRTIO_NET_CTRL_GUEST_OFFLOADS => self.handle_offloads(vdev, cmd, data),
            _ => false,
        };
        if ok { VIRTIO_NET_OK } else { VIRTIO_NET_ERR }
    }

    /// `virtio_net_handle_rx_mode()`.
    fn handle_rx_mode(&mut self, cmd: u8, data: &[u8]) -> bool {
        let Some(&on) = data.first() else {
            return false;
        };
        let on = on != 0;
        let flag = match cmd {
            VIRTIO_NET_CTRL_RX_PROMISC => &mut self.promisc,
            VIRTIO_NET_CTRL_RX_ALLMULTI => &mut self.allmulti,
            VIRTIO_NET_CTRL_RX_ALLUNI => &mut self.alluni,
            VIRTIO_NET_CTRL_RX_NOMULTI => &mut self.nomulti,
            VIRTIO_NET_CTRL_RX_NOUNI => &mut self.nouni,
            VIRTIO_NET_CTRL_RX_NOBCAST => &mut self.nobcast,
            _ => return false,
        };
        *flag = on;
        true
    }

    /// `virtio_net_handle_mac()`.
    fn handle_mac(&mut self, cmd: u8, data: &[u8]) -> bool {
        if cmd == VIRTIO_NET_CTRL_MAC_ADDR_SET {
            let Ok(mac) = <[u8; 6]>::try_from(data) else {
                return false;
            };
            self.mac = mac;
            return true;
        }
        if cmd != VIRTIO_NET_CTRL_MAC_TABLE_SET {
            return false;
        }

        let read_entries = |d: &[u8]| -> Option<u64> {
            let b = d.get(..4)?;
            Some(u64::from(u32::from_le_bytes([b[0], b[1], b[2], b[3]])))
        };
        let parse = |d: &[u8], n: usize| -> Vec<[u8; 6]> {
            d.chunks_exact(6).take(n).map(|c| [c[0], c[1], c[2], c[3], c[4], c[5]]).collect()
        };

        let mut macs = Vec::new();
        let (mut uni_overflow, mut multi_overflow) = (false, false);

        let Some(uni) = read_entries(data) else {
            return false;
        };
        let rest = &data[4..];
        if uni * 6 > rest.len() as u64 {
            return false;
        }
        let uni = uni as usize;
        if uni <= MAC_TABLE_ENTRIES {
            macs.extend(parse(rest, uni));
        } else {
            uni_overflow = true;
        }
        let rest = &rest[uni * 6..];
        let first_multi = macs.len();

        let Some(multi) = read_entries(rest) else {
            return false;
        };
        let rest = &rest[4..];
        if multi * 6 != rest.len() as u64 {
            return false;
        }
        let multi = multi as usize;
        if multi <= MAC_TABLE_ENTRIES - macs.len() {
            macs.extend(parse(rest, multi));
        } else {
            multi_overflow = true;
        }

        self.mac_table = MacTable { macs, first_multi, uni_overflow, multi_overflow };
        true
    }

    /// `virtio_net_handle_vlan_table()`.
    fn handle_vlan_table(&mut self, cmd: u8, data: &[u8]) -> bool {
        let Some(b) = data.get(..2) else {
            return false;
        };
        let vid = u16::from_le_bytes([b[0], b[1]]);
        if vid >= MAX_VLAN {
            return false;
        }
        let word = &mut self.vlans[usize::from(vid >> 5)];
        match cmd {
            VIRTIO_NET_CTRL_VLAN_ADD => *word |= 1 << (vid & 0x1f),
            VIRTIO_NET_CTRL_VLAN_DEL => *word &= !(1 << (vid & 0x1f)),
            _ => return false,
        }
        true
    }

    /// `virtio_net_handle_announce()`.
    fn handle_announce(&mut self, cmd: u8) -> bool {
        if cmd == VIRTIO_NET_CTRL_ANNOUNCE_ACK && self.status & VIRTIO_NET_S_ANNOUNCE != 0 {
            self.status &= !VIRTIO_NET_S_ANNOUNCE;
            true
        } else {
            false
        }
    }

    /// `virtio_net_handle_mq()`. RSS and hash reporting are not supported, so their commands
    /// fail the way QEMU's do when the features were not negotiated.
    fn handle_mq(&mut self, vdev: &mut VirtIODevice, cmd: u8, data: &[u8]) -> bool {
        if cmd != VIRTIO_NET_CTRL_MQ_VQ_PAIRS_SET {
            return false;
        }
        if !vdev.has_feature(VIRTIO_NET_F_MQ) {
            return false;
        }
        let Some(b) = data.get(..2) else {
            return false;
        };
        let pairs = u16::from_le_bytes([b[0], b[1]]);
        if !(VIRTIO_NET_CTRL_MQ_VQ_PAIRS_MIN..=VIRTIO_NET_CTRL_MQ_VQ_PAIRS_MAX).contains(&pairs)
            || pairs > self.max_queue_pairs
            || !self.multiqueue
        {
            return false;
        }
        self.curr_queue_pairs = pairs;
        self.update_queues(vdev, vdev.status());
        true
    }

    /// `virtio_net_handle_offloads()`.
    fn handle_offloads(&mut self, vdev: &mut VirtIODevice, cmd: u8, data: &[u8]) -> bool {
        if !vdev.has_feature(VIRTIO_NET_F_CTRL_GUEST_OFFLOADS) {
            return false;
        }
        let Some(b) = data.get(..8) else {
            return false;
        };
        if cmd != VIRTIO_NET_CTRL_GUEST_OFFLOADS_SET {
            return false;
        }
        let mut offloads = u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
        if !self.has_vnet_hdr {
            return false;
        }
        offloads &= !feature(VIRTIO_NET_F_RSC_EXT);
        let supported = vdev.guest_features() & GUEST_OFFLOADS_MASK;
        if offloads & !supported != 0 {
            return false;
        }
        self.curr_guest_offloads = offloads;
        self.apply_guest_offloads();
        true
    }
}

impl VirtioDeviceClass for VirtioNet {
    fn realize(&mut self, vdev: &mut VirtIODevice) -> Result<()> {
        let conf = &self.conf;
        let mut host_features = conf.host_features;
        if conf.host_mtu != 0 {
            host_features |= feature(VIRTIO_NET_F_MTU);
        }
        match conf.duplex.as_deref() {
            Some("half") => self.duplex = DUPLEX_HALF,
            Some("full") => self.duplex = DUPLEX_FULL,
            Some(_) => return Err(Error::generic("'duplex' must be 'half' or 'full'")),
            None => self.duplex = DUPLEX_UNKNOWN,
        }
        if self.duplex != DUPLEX_UNKNOWN {
            host_features |= feature(VIRTIO_NET_F_SPEED_DUPLEX);
        }
        if conf.speed < SPEED_UNKNOWN {
            return Err(Error::generic("'speed' must be between 0 and INT_MAX"));
        }
        if conf.speed >= 0 {
            host_features |= feature(VIRTIO_NET_F_SPEED_DUPLEX);
        }
        self.host_features = host_features;
        self.config_size = Self::config_size(host_features);

        let rx = conf.rx_queue_size;
        if !(VIRTIO_NET_RX_QUEUE_MIN_SIZE..=VIRTQUEUE_MAX_SIZE).contains(&rx)
            || !rx.is_power_of_two()
        {
            return Err(Error::generic(format!(
                "Invalid rx_queue_size (= {rx}), must be a power of 2 between \
                 {VIRTIO_NET_RX_QUEUE_MIN_SIZE} and {VIRTQUEUE_MAX_SIZE}."
            )));
        }
        let tx = conf.tx_queue_size;
        if !(VIRTIO_NET_TX_QUEUE_MIN_SIZE..=VIRTIO_NET_TX_QUEUE_MAX_SIZE).contains(&tx)
            || !tx.is_power_of_two()
        {
            return Err(Error::generic(format!(
                "Invalid tx_queue_size (= {tx}), must be a power of 2 between \
                 {VIRTIO_NET_TX_QUEUE_MIN_SIZE} and {VIRTIO_NET_TX_QUEUE_MAX_SIZE}"
            )));
        }
        let pairs = conf.queue_pairs.max(1);
        if u64::from(pairs) * 2 + 1 > VIRTIO_QUEUE_MAX as u64 {
            return Err(Error::generic(format!(
                "Invalid number of queue pairs (= {pairs}), must be a positive integer less \
                 than {}.",
                (VIRTIO_QUEUE_MAX - 1) / 2
            )));
        }
        self.max_queue_pairs = pairs as u16;
        self.curr_queue_pairs = 1;
        self.vqs = vec![NetQueue::default(); usize::from(self.max_queue_pairs)];

        vdev.init(TYPE_VIRTIO_NET, VIRTIO_ID_NET, self.config_size);
        vdev.add_queue(rx)?;
        vdev.add_queue(tx)?;
        vdev.add_queue(VIRTIO_NET_CTRL_QUEUE_SIZE)?;

        self.mac = self.conf.mac;
        self.status = VIRTIO_NET_S_LINK_UP;
        self.has_vnet_hdr = self.peer.as_ref().is_some_and(|p| p.has_vnet_hdr());
        self.host_hdr_len = if self.has_vnet_hdr { VIRTIO_NET_HDR_LEN } else { 0 };
        self.set_mrg_rx_bufs(false, false);
        self.promisc = true;
        self.mac_table = MacTable::default();
        self.vlans.fill(u32::MAX);
        Ok(())
    }

    fn get_features(&mut self, _vdev: &VirtIODevice, features: u64) -> Result<u64> {
        let mut f = features | self.host_features | feature(VIRTIO_NET_F_MAC);
        let (vnet_hdr, ufo, uso) = match self.peer.as_ref() {
            Some(p) => (p.has_vnet_hdr(), p.has_ufo(), p.has_uso()),
            None => (false, false, false),
        };
        if !vnet_hdr {
            for bit in [
                VIRTIO_NET_F_CSUM,
                VIRTIO_NET_F_HOST_TSO4,
                VIRTIO_NET_F_HOST_TSO6,
                VIRTIO_NET_F_HOST_ECN,
                VIRTIO_NET_F_GUEST_CSUM,
                VIRTIO_NET_F_GUEST_TSO4,
                VIRTIO_NET_F_GUEST_TSO6,
                VIRTIO_NET_F_GUEST_ECN,
                VIRTIO_NET_F_HOST_USO,
                VIRTIO_NET_F_GUEST_USO4,
                VIRTIO_NET_F_GUEST_USO6,
                VIRTIO_NET_F_HASH_REPORT,
            ] {
                f &= !feature(bit);
            }
        }
        if !vnet_hdr || !ufo {
            f &= !(feature(VIRTIO_NET_F_GUEST_UFO) | feature(VIRTIO_NET_F_HOST_UFO));
        }
        if !uso {
            f &= !(feature(VIRTIO_NET_F_HOST_USO)
                | feature(VIRTIO_NET_F_GUEST_USO4)
                | feature(VIRTIO_NET_F_GUEST_USO6));
        }
        // Without an eBPF steering program there is no RSS or hash reporting.
        f &= !(feature(VIRTIO_NET_F_RSS) | feature(VIRTIO_NET_F_HASH_REPORT));
        Ok(f)
    }

    fn set_features(&mut self, vdev: &mut VirtIODevice, features: u64) {
        self.set_multiqueue(
            vdev,
            has_feature(features, VIRTIO_NET_F_RSS) || has_feature(features, VIRTIO_NET_F_MQ),
        );
        self.set_mrg_rx_bufs(
            has_feature(features, VIRTIO_NET_F_MRG_RXBUF),
            has_feature(features, VIRTIO_F_VERSION_1),
        );
        if self.has_vnet_hdr {
            self.curr_guest_offloads = features & GUEST_OFFLOADS_MASK;
            self.apply_guest_offloads();
        }
        // The guest features still hold the old value here.
        let vlan = has_feature(features, VIRTIO_NET_F_CTRL_VLAN);
        if vlan != vdev.has_feature(VIRTIO_NET_F_CTRL_VLAN) {
            self.vlans.fill(if vlan { 0 } else { u32::MAX });
        }
    }

    fn get_config(&mut self, vdev: &VirtIODevice, config: &mut [u8]) {
        let mut c = [0u8; VIRTIO_NET_CONFIG_SIZE];
        c[CFG_MAC..CFG_MAC + 6].copy_from_slice(&self.mac);
        c[CFG_STATUS..CFG_STATUS + 2].copy_from_slice(&self.status.to_le_bytes());
        c[CFG_MAX_VQ_PAIRS..CFG_MAX_VQ_PAIRS + 2]
            .copy_from_slice(&self.max_queue_pairs.to_le_bytes());
        c[CFG_MTU..CFG_MTU + 2].copy_from_slice(&self.conf.host_mtu.to_le_bytes());
        c[CFG_SPEED..CFG_SPEED + 4].copy_from_slice(&self.conf.speed.to_le_bytes());
        c[CFG_DUPLEX] = self.duplex;
        c[CFG_RSS_MAX_KEY_SIZE] = VIRTIO_NET_RSS_MAX_KEY_SIZE;
        let table_len: u16 = if vdev.host_has_feature(VIRTIO_NET_F_RSS) { 128 } else { 1 };
        c[CFG_RSS_MAX_INDIRECTION_TABLE_LENGTH..CFG_RSS_MAX_INDIRECTION_TABLE_LENGTH + 2]
            .copy_from_slice(&table_len.to_le_bytes());
        c[CFG_SUPPORTED_HASH_TYPES..CFG_SUPPORTED_HASH_TYPES + 4].fill(0);
        let n = config.len().min(self.config_size);
        config[..n].copy_from_slice(&c[..n]);
    }

    fn set_config(&mut self, vdev: &mut VirtIODevice, config: &mut [u8]) {
        let Some(mac) = config.get(CFG_MAC..CFG_MAC + 6) else {
            return;
        };
        if !vdev.has_feature(VIRTIO_NET_F_CTRL_MAC_ADDR)
            && !vdev.has_feature(VIRTIO_F_VERSION_1)
            && mac != self.mac
        {
            self.mac.copy_from_slice(mac);
        }
    }

    fn set_status(&mut self, vdev: &mut VirtIODevice, status: u8) -> Result<()> {
        self.update_queues(vdev, status);
        Ok(())
    }

    fn reset(&mut self, _vdev: &mut VirtIODevice) {
        self.promisc = true;
        self.allmulti = false;
        self.alluni = false;
        self.nomulti = false;
        self.nouni = false;
        self.nobcast = false;
        self.curr_queue_pairs = 1;
        self.status &= !VIRTIO_NET_S_ANNOUNCE;
        self.mac_table = MacTable::default();
        self.mac = self.conf.mac;
        for q in &mut self.vqs {
            q.tx_waiting = false;
        }
    }

    fn handle_output(&mut self, vdev: &mut VirtIODevice, queue: u16) {
        if queue == Self::ctrl_queue(vdev) {
            self.handle_ctrl(vdev, queue);
        } else if queue % 2 == 0 {
            if let Some(peer) = self.peer.as_mut() {
                peer.rx_ready(queue / 2);
            }
        } else {
            self.handle_tx(vdev, queue / 2);
        }
    }

    fn legacy_features(&self) -> u64 {
        VIRTIO_LEGACY_FEATURES | feature(VIRTIO_NET_F_GSO)
    }

    fn pre_load_queues(&mut self, vdev: &mut VirtIODevice, n: usize) -> Result<()> {
        self.vmstate_pre_load_queues(vdev, n)
    }

    fn post_load(&mut self, vdev: &mut VirtIODevice) -> Result<()> {
        self.vmstate_post_load(vdev)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
