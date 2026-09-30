// SPDX-License-Identifier: GPL-2.0-or-later

//! The net client model of net/net.c and the packet queue of net/queue.c.
//!
//! A [`NetClient`] is one end of a link: a NIC model, a backend such as tap, or a hub port. Two
//! clients are joined as peers. Sending on a client puts the packet on its peer's incoming queue,
//! which hands it to the peer's [`NetClientOps::receive`] right away when it can, and holds on to
//! it otherwise.
//!
//! QEMU runs all of this under the big lock. Here backends read on threads of their own, so every
//! piece of shared state has a small lock of its own, and no lock is ever held while calling into
//! [`NetClientOps`] or a [`SentCb`]. The queue's `delivering` flag, which QEMU uses to catch
//! re-entry on one thread, also keeps two threads from delivering into one queue at once.

use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};

use ruvm_qapi::types::NetClientDriver;

use crate::hub::Hub;

/// `NET_BUFSIZE`: the largest packet a backend reads in one go.
pub const NET_BUFSIZE: usize = 4096 + 65536;

/// `ETH_ZLEN`: the minimum Ethernet frame length without the FCS.
pub const ETH_ZLEN: usize = 60;

/// `MAX_QUEUE_NUM`: the most queues one multiqueue client can have.
pub const MAX_QUEUE_NUM: usize = 1024;

/// The most packets a queue holds for senders that have no completion callback.
pub const QUEUE_MAXLEN: usize = 10000;

/// `QEMU_NET_PACKET_FLAG_NONE`.
pub const PACKET_FLAG_NONE: u32 = 0;

/// `QEMU_NET_PACKET_FLAG_RAW`: the packet has no virtio-net header even if the receiver uses one,
/// so an empty header gets put in front of it on delivery.
pub const PACKET_FLAG_RAW: u32 = 1;

/// The size of `struct virtio_net_hdr`.
pub const VNET_HDR_LEN: usize = 10;
/// The size of `struct virtio_net_hdr_mrg_rxbuf`.
pub const VNET_HDR_MRG_RXBUF_LEN: usize = 12;
/// The size of `struct virtio_net_hdr_v1_hash`.
pub const VNET_HDR_V1_HASH_LEN: usize = 20;
/// The size of `struct virtio_net_hdr_v1_hash_tunnel`.
pub const VNET_HDR_V1_HASH_TUNNEL_LEN: usize = 24;

/// `NetPacketSent`: told how a queued packet ended. The argument is what the receiver returned,
/// or 0 when the packet was purged without being delivered.
pub type SentCb = Box<dyn FnOnce(isize) + Send>;

/// `NetOffloads`: the offloads a NIC asks its backend to turn on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NetOffloads {
    pub csum: bool,
    pub tso4: bool,
    pub tso6: bool,
    pub ecn: bool,
    pub ufo: bool,
    pub uso4: bool,
    pub uso6: bool,
    pub tnl: bool,
    pub tnl_csum: bool,
}

/// `NetClientInfo`: what a NIC model or a backend does with the packets and requests that reach
/// it. Everything but [`NetClientOps::receive`] is optional, as in QEMU.
///
/// The methods are called from whatever thread sends or flushes, so implementations must be
/// thread safe. No lock of this crate is held during the calls, so they may send packets of their
/// own.
pub trait NetClientOps: Send + Sync {
    /// Takes one packet, given as the pieces of a scatter list. Returns the number of bytes taken,
    /// 0 to have the packet queued until [`NetClient::flush_queued_packets`] is called, or a
    /// negative errno.
    fn receive(&self, nc: &NetClient, iov: &[&[u8]]) -> isize;

    fn can_receive(&self, nc: &NetClient) -> bool {
        let _ = nc;
        true
    }

    fn link_status_changed(&self, nc: &NetClient) {
        let _ = nc;
    }

    /// Called once when the client is deleted.
    fn cleanup(&self, nc: &NetClient) {
        let _ = nc;
    }

    /// `poll`: stops or restarts reading from the host side.
    fn poll(&self, nc: &NetClient, enable: bool) {
        let _ = (nc, enable);
    }

    fn has_ufo(&self, nc: &NetClient) -> bool {
        let _ = nc;
        false
    }

    fn has_uso(&self, nc: &NetClient) -> bool {
        let _ = nc;
        false
    }

    fn has_tunnel(&self, nc: &NetClient) -> bool {
        let _ = nc;
        false
    }

    fn has_vnet_hdr(&self, nc: &NetClient) -> bool {
        let _ = nc;
        false
    }

    fn has_vnet_hdr_len(&self, nc: &NetClient, len: usize) -> bool {
        let _ = (nc, len);
        false
    }

    /// Switches to virtio-net headers of `len` bytes. Returns false if the client has no such
    /// operation, in which case the length is not recorded either.
    fn set_vnet_hdr_len(&self, nc: &NetClient, len: usize) -> bool {
        let _ = (nc, len);
        false
    }

    fn set_offload(&self, nc: &NetClient, ol: &NetOffloads) {
        let _ = (nc, ol);
    }

    /// `set_vnet_le`. `None` means the client has no such operation.
    fn set_vnet_le(&self, nc: &NetClient, is_le: bool) -> Option<std::io::Result<()>> {
        let _ = (nc, is_le);
        None
    }

    /// `set_vnet_be`. `None` means the client has no such operation.
    fn set_vnet_be(&self, nc: &NetClient, is_be: bool) -> Option<std::io::Result<()>> {
        let _ = (nc, is_be);
        None
    }
}

pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn iov_size(iov: &[&[u8]]) -> usize {
    iov.iter().map(|b| b.len()).sum()
}

fn iov_to_vec(iov: &[&[u8]]) -> Vec<u8> {
    let mut v = Vec::with_capacity(iov_size(iov));
    for b in iov {
        v.extend_from_slice(b);
    }
    v
}

/// `eth_pad_short_frame()`: `pkt` padded with zeroes to [`ETH_ZLEN`], or `None` if it is long
/// enough already.
pub fn eth_pad_short_frame(pkt: &[u8]) -> Option<[u8; ETH_ZLEN]> {
    if pkt.len() >= ETH_ZLEN {
        return None;
    }
    let mut padded = [0u8; ETH_ZLEN];
    padded[..pkt.len()].copy_from_slice(pkt);
    Some(padded)
}

struct NetPacket {
    sender: Weak<NetClient>,
    flags: u32,
    data: Vec<u8>,
    sent_cb: Option<SentCb>,
}

struct QueueInner {
    packets: VecDeque<NetPacket>,
    delivering: bool,
    /// Someone asked for a flush while a delivery was running. If that delivery is refused, it is
    /// tried once more instead of disabling the receiver, since the receiver may have become ready
    /// in between.
    retry: bool,
    /// The client's `receive_disabled`, kept here so that it changes together with the queue.
    receive_disabled: bool,
    maxlen: usize,
}

/// `NetQueue`: the packets waiting for a client to be able to receive.
pub struct NetQueue {
    inner: Mutex<QueueInner>,
}

impl fmt::Debug for NetQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let g = lock(&self.inner);
        f.debug_struct("NetQueue")
            .field("len", &g.packets.len())
            .field("delivering", &g.delivering)
            .field("receive_disabled", &g.receive_disabled)
            .finish()
    }
}

impl NetQueue {
    fn new() -> Self {
        NetQueue {
            inner: Mutex::new(QueueInner {
                packets: VecDeque::new(),
                delivering: false,
                retry: false,
                receive_disabled: false,
                maxlen: QUEUE_MAXLEN,
            }),
        }
    }

    /// The number of packets waiting.
    pub fn len(&self) -> usize {
        lock(&self.inner).packets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Changes `nq_maxlen`, mostly for tests.
    pub fn set_maxlen(&self, maxlen: usize) {
        lock(&self.inner).maxlen = maxlen;
    }

    /// `qemu_net_queue_append_iov()`: queues a copy, or drops it when the queue is full and nobody
    /// would be told.
    fn append(
        g: &mut QueueInner,
        sender: Weak<NetClient>,
        flags: u32,
        iov: &[&[u8]],
        sent_cb: Option<SentCb>,
    ) {
        if g.packets.len() >= g.maxlen && sent_cb.is_none() {
            return;
        }
        g.packets.push_back(NetPacket { sender, flags, data: iov_to_vec(iov), sent_cb });
    }

    /// Delivers one packet with `delivering` already set. Returns the relocked queue and the
    /// receiver's answer. This is `qemu_deliver_packet_iov()` with the `receive_disabled` part done
    /// under the queue lock.
    fn deliver_one<'a>(
        &'a self,
        owner: &NetClient,
        mut g: MutexGuard<'a, QueueInner>,
        flags: u32,
        iov: &[&[u8]],
    ) -> (MutexGuard<'a, QueueInner>, isize) {
        loop {
            let disabled = g.receive_disabled;
            drop(g);
            let ret = if owner.link_down() {
                iov_size(iov) as isize
            } else if disabled {
                0
            } else {
                owner.deliver(flags, iov)
            };
            g = lock(&self.inner);
            if ret != 0 {
                return (g, ret);
            }
            if g.retry {
                // Somebody flushed while this delivery was on its way, so the receiver may be
                // ready now. Try once more instead of disabling it.
                g.retry = false;
                continue;
            }
            g.receive_disabled = true;
            return (g, 0);
        }
    }

    fn end_delivering(g: &mut QueueInner) {
        g.delivering = false;
        g.retry = false;
    }

    /// `qemu_net_queue_send_iov()`.
    fn send(
        &self,
        owner: &NetClient,
        sender: &NetClient,
        flags: u32,
        iov: &[&[u8]],
        sent_cb: Option<SentCb>,
    ) -> isize {
        let can_send = sender.can_send_packet();
        let mut g = lock(&self.inner);
        if g.delivering || !can_send {
            Self::append(&mut g, sender.this.clone(), flags, iov, sent_cb);
            return 0;
        }
        g.delivering = true;
        let (mut g, ret) = self.deliver_one(owner, g, flags, iov);
        if ret == 0 {
            Self::append(&mut g, sender.this.clone(), flags, iov, sent_cb);
            Self::end_delivering(&mut g);
            return 0;
        }
        self.flush_locked(owner, g);
        ret
    }

    /// `qemu_net_queue_receive()`: delivery with no sender, for loopback.
    fn receive(&self, owner: &NetClient, data: &[u8]) -> isize {
        let mut g = lock(&self.inner);
        if g.delivering {
            return 0;
        }
        g.delivering = true;
        let (mut g, ret) = self.deliver_one(owner, g, PACKET_FLAG_NONE, &[data]);
        Self::end_delivering(&mut g);
        ret
    }

    /// `qemu_net_queue_purge()`: drops the packets `from` sent, telling each sender.
    pub fn purge(&self, from: &NetClient) {
        let target: *const NetClient = from;
        let purged: Vec<NetPacket> = {
            let mut g = lock(&self.inner);
            let (out, keep): (Vec<_>, Vec<_>) =
                g.packets.drain(..).partition(|p| std::ptr::eq(p.sender.as_ptr(), target));
            g.packets = keep.into();
            out
        };
        for p in purged {
            if let Some(cb) = p.sent_cb {
                cb(0);
            }
        }
    }

    /// `qemu_net_queue_flush()`: delivers what it can. True when the queue ended up empty.
    fn flush(&self, owner: &NetClient) -> bool {
        let mut g = lock(&self.inner);
        if g.delivering {
            g.retry = true;
            return false;
        }
        g.delivering = true;
        self.flush_locked(owner, g)
    }

    fn flush_locked<'a>(&'a self, owner: &NetClient, mut g: MutexGuard<'a, QueueInner>) -> bool {
        loop {
            let Some(p) = g.packets.pop_front() else {
                Self::end_delivering(&mut g);
                return true;
            };
            let (g2, ret) = self.deliver_one(owner, g, p.flags, &[&p.data]);
            g = g2;
            if ret == 0 {
                g.packets.push_front(p);
                Self::end_delivering(&mut g);
                return false;
            }
            if let Some(cb) = p.sent_cb {
                drop(g);
                cb(ret);
                g = lock(&self.inner);
            }
        }
    }
}

/// `NetClientState`.
pub struct NetClient {
    this: Weak<NetClient>,
    driver: NetClientDriver,
    model: String,
    name: String,
    queue_index: AtomicU32,
    info_str: Mutex<String>,
    peer: Mutex<Weak<NetClient>>,
    link_down: AtomicBool,
    vnet_hdr_len: AtomicUsize,
    is_netdev: AtomicBool,
    do_not_pad: AtomicBool,
    incoming: NetQueue,
    ops: Arc<dyn NetClientOps>,
    vm_running: Arc<AtomicBool>,
    pub(crate) hub: OnceLock<(Weak<Hub>, usize)>,
}

impl fmt::Debug for NetClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NetClient")
            .field("name", &self.name)
            .field("model", &self.model)
            .field("driver", &self.driver)
            .field("link_down", &self.link_down())
            .finish_non_exhaustive()
    }
}

impl NetClient {
    /// Makes a client that is not in any [`crate::Net`] and has no peer. `make_ops` gets a weak
    /// reference to the new client so that the operations can find it again.
    ///
    /// Most code wants [`crate::Net::new_client`] instead, which names the client the way QEMU
    /// does and keeps it in the list `info network` shows.
    pub fn new(
        driver: NetClientDriver,
        model: &str,
        name: &str,
        make_ops: impl FnOnce(&Weak<NetClient>) -> Arc<dyn NetClientOps>,
    ) -> Arc<NetClient> {
        Self::with_runstate(driver, model, name, Arc::new(AtomicBool::new(true)), make_ops)
    }

    pub(crate) fn with_runstate(
        driver: NetClientDriver,
        model: &str,
        name: &str,
        vm_running: Arc<AtomicBool>,
        make_ops: impl FnOnce(&Weak<NetClient>) -> Arc<dyn NetClientOps>,
    ) -> Arc<NetClient> {
        Arc::new_cyclic(|this| NetClient {
            this: this.clone(),
            driver,
            model: model.to_string(),
            name: name.to_string(),
            queue_index: AtomicU32::new(0),
            info_str: Mutex::new(String::new()),
            peer: Mutex::new(Weak::new()),
            link_down: AtomicBool::new(false),
            vnet_hdr_len: AtomicUsize::new(0),
            is_netdev: AtomicBool::new(false),
            do_not_pad: AtomicBool::new(false),
            incoming: NetQueue::new(),
            ops: make_ops(this),
            vm_running,
            hub: OnceLock::new(),
        })
    }

    /// Joins two clients that have no peer yet.
    pub fn connect(a: &Arc<NetClient>, b: &Arc<NetClient>) {
        assert!(a.peer().is_none() && b.peer().is_none(), "net client already has a peer");
        *lock(&a.peer) = Arc::downgrade(b);
        *lock(&b.peer) = Arc::downgrade(a);
    }

    pub(crate) fn clear_peer(&self) {
        *lock(&self.peer) = Weak::new();
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    /// `nc->info->type`.
    pub fn driver(&self) -> NetClientDriver {
        self.driver
    }

    pub fn ops(&self) -> &Arc<dyn NetClientOps> {
        &self.ops
    }

    pub fn queue_index(&self) -> u32 {
        self.queue_index.load(Ordering::Relaxed)
    }

    pub fn set_queue_index(&self, index: u32) {
        self.queue_index.store(index, Ordering::Relaxed);
    }

    /// `nc->info_str`, the text `info network` shows after the type.
    pub fn info_str(&self) -> String {
        lock(&self.info_str).clone()
    }

    /// `qemu_set_info_str()`. QEMU keeps at most 255 bytes, and so does this.
    pub fn set_info_str(&self, s: &str) {
        let mut end = s.len().min(255);
        while !s.is_char_boundary(end) {
            end -= 1;
        }
        *lock(&self.info_str) = s[..end].to_string();
    }

    pub fn peer(&self) -> Option<Arc<NetClient>> {
        lock(&self.peer).upgrade()
    }

    pub fn link_down(&self) -> bool {
        self.link_down.load(Ordering::SeqCst)
    }

    /// Sets `link_down` without telling anybody. [`crate::Net::set_link`] is `set_link`.
    pub fn set_link_down(&self, down: bool) {
        self.link_down.store(down, Ordering::SeqCst);
    }

    pub fn receive_disabled(&self) -> bool {
        lock(&self.incoming.inner).receive_disabled
    }

    pub fn is_netdev(&self) -> bool {
        self.is_netdev.load(Ordering::Relaxed)
    }

    pub(crate) fn set_is_netdev(&self) {
        self.is_netdev.store(true, Ordering::Relaxed);
    }

    /// `do_not_pad`: NIC models that cope with short frames set this.
    pub fn set_do_not_pad(&self, v: bool) {
        self.do_not_pad.store(v, Ordering::Relaxed);
    }

    /// `net_peer_needs_padding()`.
    pub fn peer_needs_padding(&self) -> bool {
        self.peer().is_some_and(|p| !p.do_not_pad.load(Ordering::Relaxed))
    }

    /// `incoming_queue`.
    pub fn incoming_queue(&self) -> &NetQueue {
        &self.incoming
    }

    /// `qemu_can_receive_packet()`.
    pub fn can_receive_packet(&self) -> bool {
        !self.receive_disabled() && self.ops.can_receive(self)
    }

    /// `qemu_can_send_packet()`: false while the VM is stopped, true without a peer.
    pub fn can_send_packet(&self) -> bool {
        if !self.vm_running.load(Ordering::SeqCst) {
            return false;
        }
        match self.peer() {
            None => true,
            Some(peer) => peer.can_receive_packet(),
        }
    }

    /// The part of `qemu_deliver_packet_iov()` that calls the receiver.
    fn deliver(&self, flags: u32, iov: &[&[u8]]) -> isize {
        let hdr_len = self.vnet_hdr_len();
        if flags & PACKET_FLAG_RAW != 0 && hdr_len != 0 {
            let hdr = [0u8; VNET_HDR_V1_HASH_TUNNEL_LEN];
            let mut v: Vec<&[u8]> = Vec::with_capacity(iov.len() + 1);
            v.push(&hdr[..hdr_len.min(hdr.len())]);
            v.extend_from_slice(iov);
            return self.ops.receive(self, &v);
        }
        self.ops.receive(self, iov)
    }

    fn send_with_flags(&self, flags: u32, iov: &[&[u8]], sent_cb: Option<SentCb>) -> isize {
        let size = iov_size(iov) as isize;
        if self.link_down() {
            return size;
        }
        let Some(peer) = self.peer() else {
            return size;
        };
        peer.incoming.send(&peer, self, flags, iov, sent_cb)
    }

    /// `qemu_send_packet_async()`. Returns 0 when the packet got queued, in which case `sent_cb`
    /// runs once it leaves the queue.
    pub fn send_packet_async(&self, buf: &[u8], sent_cb: Option<SentCb>) -> isize {
        self.send_with_flags(PACKET_FLAG_NONE, &[buf], sent_cb)
    }

    /// `qemu_send_packet()`.
    pub fn send_packet(&self, buf: &[u8]) -> isize {
        self.send_packet_async(buf, None)
    }

    /// `qemu_send_packet_raw()`.
    pub fn send_packet_raw(&self, buf: &[u8]) -> isize {
        self.send_with_flags(PACKET_FLAG_RAW, &[buf], None)
    }

    /// `qemu_sendv_packet_async()`.
    pub fn sendv_packet_async(&self, iov: &[&[u8]], sent_cb: Option<SentCb>) -> isize {
        let size = iov_size(iov);
        if size > NET_BUFSIZE {
            return size as isize;
        }
        self.send_with_flags(PACKET_FLAG_NONE, iov, sent_cb)
    }

    /// `qemu_sendv_packet()`.
    pub fn sendv_packet(&self, iov: &[&[u8]]) -> isize {
        self.sendv_packet_async(iov, None)
    }

    /// `qemu_receive_packet()`: loops a packet back into this client.
    pub fn receive_packet(&self, buf: &[u8]) -> isize {
        if !self.can_receive_packet() {
            return 0;
        }
        if !self.do_not_pad.load(Ordering::Relaxed) {
            if let Some(padded) = eth_pad_short_frame(buf) {
                return self.incoming.receive(self, &padded);
            }
        }
        self.incoming.receive(self, buf)
    }

    /// `qemu_purge_queued_packets()`: drops what this client queued at its peer.
    pub fn purge_queued_packets(&self) {
        if let Some(peer) = self.peer() {
            peer.incoming.purge(self);
        }
    }

    /// `qemu_flush_or_purge_queued_packets()`.
    pub fn flush_or_purge_queued_packets(&self, purge: bool) {
        {
            let mut g = lock(&self.incoming.inner);
            g.receive_disabled = false;
            if g.delivering {
                g.retry = true;
            }
        }
        if let Some(peer) = self.peer() {
            if peer.driver == NetClientDriver::Hubport {
                crate::hub::flush(&peer);
            }
        }
        if !self.incoming.flush(self) && purge {
            if let Some(peer) = self.peer() {
                self.incoming.purge(&peer);
            }
        }
    }

    /// `qemu_flush_queued_packets()`: to be called by a client that can receive again.
    pub fn flush_queued_packets(&self) {
        self.flush_or_purge_queued_packets(false);
    }

    pub(crate) fn flush_incoming(&self) -> bool {
        self.incoming.flush(self)
    }

    pub fn has_ufo(&self) -> bool {
        self.ops.has_ufo(self)
    }

    pub fn has_uso(&self) -> bool {
        self.ops.has_uso(self)
    }

    pub fn has_tunnel(&self) -> bool {
        self.ops.has_tunnel(self)
    }

    pub fn has_vnet_hdr(&self) -> bool {
        self.ops.has_vnet_hdr(self)
    }

    pub fn has_vnet_hdr_len(&self, len: usize) -> bool {
        self.ops.has_vnet_hdr_len(self, len)
    }

    /// `qemu_set_offload()`.
    pub fn set_offload(&self, ol: &NetOffloads) {
        self.ops.set_offload(self, ol);
    }

    /// `qemu_get_vnet_hdr_len()`.
    pub fn vnet_hdr_len(&self) -> usize {
        self.vnet_hdr_len.load(Ordering::Relaxed)
    }

    /// `qemu_set_vnet_hdr_len()`. The length must be one of the virtio-net header sizes.
    pub fn set_vnet_hdr_len(&self, len: usize) {
        assert!(
            matches!(
                len,
                VNET_HDR_LEN
                    | VNET_HDR_MRG_RXBUF_LEN
                    | VNET_HDR_V1_HASH_LEN
                    | VNET_HDR_V1_HASH_TUNNEL_LEN
            ),
            "bad vnet header length {len}"
        );
        if self.ops.set_vnet_hdr_len(self, len) {
            self.vnet_hdr_len.store(len, Ordering::Relaxed);
        }
    }

    /// `qemu_set_vnet_le()`: nothing to do on a little-endian host.
    pub fn set_vnet_le(&self, is_le: bool) -> std::io::Result<()> {
        if cfg!(target_endian = "big") {
            return self.ops.set_vnet_le(self, is_le).unwrap_or_else(|| Err(unsupported()));
        }
        Ok(())
    }

    /// `qemu_set_vnet_be()`: nothing to do on a big-endian host.
    pub fn set_vnet_be(&self, is_be: bool) -> std::io::Result<()> {
        if cfg!(target_endian = "little") {
            return self.ops.set_vnet_be(self, is_be).unwrap_or_else(|| Err(unsupported()));
        }
        Ok(())
    }
}

fn unsupported() -> std::io::Error {
    std::io::Error::from(std::io::ErrorKind::Unsupported)
}
