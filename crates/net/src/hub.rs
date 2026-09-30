// SPDX-License-Identifier: GPL-2.0-or-later

//! Hubs, net/hub.c.
//!
//! A hub is a dumb switch: whatever reaches one port goes out of every other port. The old
//! `-net` options put all their clients on hubs, and `-netdev hubport` adds a port by hand.

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

use ruvm_qapi::types::NetClientDriver;

use crate::client::{NetClient, NetClientOps, lock};

/// `NetHub`.
pub struct Hub {
    id: i32,
    num_ports: AtomicUsize,
    /// The ports, newest first, as QEMU's list head insertion leaves them.
    ports: Mutex<Vec<Weak<NetClient>>>,
}

impl fmt::Debug for Hub {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Hub").field("id", &self.id).field("ports", &self.ports().len()).finish()
    }
}

impl Hub {
    pub(crate) fn new(id: i32) -> Arc<Hub> {
        Arc::new(Hub { id, num_ports: AtomicUsize::new(0), ports: Mutex::new(Vec::new()) })
    }

    pub fn id(&self) -> i32 {
        self.id
    }

    /// The ports that still exist, newest first.
    pub fn ports(&self) -> Vec<Arc<NetClient>> {
        lock(&self.ports).iter().filter_map(Weak::upgrade).collect()
    }

    /// Takes the next port number and the default name that goes with it.
    pub(crate) fn next_port(&self) -> (usize, String) {
        let id = self.num_ports.fetch_add(1, Ordering::SeqCst);
        (id, format!("hub{}port{}", self.id, id))
    }

    pub(crate) fn list_port(&self, port: &Arc<NetClient>) {
        lock(&self.ports).insert(0, Arc::downgrade(port));
    }

    fn others(&self, source: &NetClient) -> Vec<Arc<NetClient>> {
        self.ports().into_iter().filter(|p| !std::ptr::eq(p.as_ref(), source)).collect()
    }
}

/// The operations of a hub port.
pub(crate) struct HubPortOps {
    pub(crate) hub: Weak<Hub>,
}

impl HubPortOps {
    fn hub(&self) -> Option<Arc<Hub>> {
        self.hub.upgrade()
    }
}

impl NetClientOps for HubPortOps {
    /// `net_hub_port_receive_iov()`.
    fn receive(&self, nc: &NetClient, iov: &[&[u8]]) -> isize {
        let len: usize = iov.iter().map(|b| b.len()).sum();
        if let Some(hub) = self.hub() {
            for port in hub.others(nc) {
                port.sendv_packet(iov);
            }
        }
        len as isize
    }

    /// `net_hub_port_can_receive()`.
    fn can_receive(&self, nc: &NetClient) -> bool {
        self.hub().is_some_and(|hub| hub.others(nc).iter().any(|p| p.can_send_packet()))
    }

    /// `net_hub_port_cleanup()`.
    fn cleanup(&self, nc: &NetClient) {
        if let Some(hub) = self.hub() {
            lock(&hub.ports).retain(|p| !std::ptr::eq(p.as_ptr(), nc));
        }
    }
}

/// `net_hub_flush()`: flushes the queues of the other ports of `port`'s hub.
pub(crate) fn flush(port: &NetClient) -> bool {
    let Some(hub) = port.hub.get().and_then(|(h, _)| h.upgrade()) else {
        return false;
    };
    let mut ret = false;
    for p in hub.others(port) {
        ret |= p.flush_incoming();
    }
    ret
}

/// `net_hub_id_for_client()`: the hub `nc` is a port of, or is plugged into.
pub fn hub_id_for_client(nc: &NetClient) -> Option<i32> {
    let hub_of = |c: &NetClient| c.hub.get().and_then(|(h, _)| h.upgrade()).map(|h| h.id());
    if nc.driver() == NetClientDriver::Hubport {
        return hub_of(nc);
    }
    let peer = nc.peer()?;
    if peer.driver() == NetClientDriver::Hubport {
        return hub_of(&peer);
    }
    None
}
