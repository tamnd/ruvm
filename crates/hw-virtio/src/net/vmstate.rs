// SPDX-License-Identifier: GPL-2.0-or-later

//! The `virtio-net-device` description: the receive filter, the queue pairs and the offloads.

use ruvm_base::{Error, Result};

use super::{
    GUEST_OFFLOADS_MASK, MAC_TABLE_ENTRIES, MacTable, VIRTIO_NET_F_CTRL_GUEST_OFFLOADS, VirtioNet,
};
use crate::virtio::{VIRTIO_F_VERSION_1, VirtIODevice};

/// `VirtIONet` as `virtio-net-device` version 11 has it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VirtioNetVmState {
    pub mac: [u8; 6],
    /// `tx_waiting` of every queue pair, [`max_queue_pairs`](Self::max_queue_pairs) entries.
    /// The first goes in the main fields, the next `curr_queue_pairs - 1` at the end.
    pub tx_waiting: Vec<u32>,
    pub mergeable_rx_bufs: u32,
    pub status: u16,
    pub promisc: u8,
    pub allmulti: u8,
    /// `mac_table.macs`, `mac_table.in_use` entries.
    pub macs: Vec<[u8; 6]>,
    /// The VLAN filter bitmap, `MAX_VLAN >> 5` words sent as host order bytes.
    pub vlans: Vec<u32>,
    pub has_vnet_hdr: u32,
    pub multi_overflow: u8,
    pub uni_overflow: u8,
    pub alluni: u8,
    pub nomulti: u8,
    pub nouni: u8,
    pub nobcast: u8,
    pub has_ufo: u8,
    /// Sent only when greater than 1, and then it must match.
    pub max_queue_pairs: u16,
    /// Sent only when [`max_queue_pairs`](Self::max_queue_pairs) is greater than 1.
    pub curr_queue_pairs: u16,
    /// Sent only when the driver has `VIRTIO_NET_F_CTRL_GUEST_OFFLOADS`.
    pub curr_guest_offloads: u64,
}

fn load_error(msg: impl std::fmt::Display) -> Error {
    Error::generic(format!("virtio-net: {msg}"))
}

impl VirtioNet {
    /// The device model's part of the migration stream.
    pub fn vmstate_save(&self) -> VirtioNetVmState {
        let b = u8::from;
        VirtioNetVmState {
            mac: self.mac,
            tx_waiting: self.vqs.iter().map(|q| u32::from(q.tx_waiting)).collect(),
            mergeable_rx_bufs: u32::from(self.mergeable_rx_bufs),
            status: self.status,
            promisc: b(self.promisc),
            allmulti: b(self.allmulti),
            macs: self.mac_table.macs.clone(),
            vlans: self.vlans.clone(),
            has_vnet_hdr: u32::from(self.has_vnet_hdr),
            multi_overflow: b(self.mac_table.multi_overflow),
            uni_overflow: b(self.mac_table.uni_overflow),
            alluni: b(self.alluni),
            nomulti: b(self.nomulti),
            nouni: b(self.nouni),
            nobcast: b(self.nobcast),
            has_ufo: b(self.peer.as_ref().is_some_and(|p| p.has_ufo())),
            max_queue_pairs: self.max_queue_pairs,
            curr_queue_pairs: self.curr_queue_pairs,
            curr_guest_offloads: self.curr_guest_offloads,
        }
    }

    /// Takes back what [`vmstate_save`](Self::vmstate_save) returned, the fields and
    /// `virtio_net_post_load_device()`. The offloads take effect in the class `post_load`, once
    /// the features are back.
    pub fn vmstate_load(&mut self, vdev: &mut VirtIODevice, s: &VirtioNetVmState) -> Result<()> {
        let peer_vnet_hdr = self.peer.as_ref().is_some_and(|p| p.has_vnet_hdr());
        if s.has_vnet_hdr != 0 && !peer_vnet_hdr {
            return Err(load_error("saved image requires vnet_hdr=on"));
        }
        if s.has_ufo != 0 && !self.peer.as_ref().is_some_and(|p| p.has_ufo()) {
            return Err(load_error("saved image requires TUN_F_UFO support"));
        }
        if self.max_queue_pairs > 1 {
            if s.max_queue_pairs != self.max_queue_pairs {
                return Err(load_error(format!(
                    "{} != {} queue pairs",
                    s.max_queue_pairs, self.max_queue_pairs
                )));
            }
            if s.curr_queue_pairs > self.max_queue_pairs || s.curr_queue_pairs == 0 {
                return Err(load_error(format!(
                    "curr_queue_pairs {} > max_queue_pairs {}",
                    s.curr_queue_pairs, self.max_queue_pairs
                )));
            }
            self.curr_queue_pairs = s.curr_queue_pairs;
        } else {
            self.curr_queue_pairs = 1;
        }
        if s.vlans.len() != self.vlans.len() {
            return Err(load_error("VLAN table size differs"));
        }

        self.mac = s.mac;
        for (i, q) in self.vqs.iter_mut().enumerate() {
            q.tx_waiting = i < usize::from(self.curr_queue_pairs)
                && s.tx_waiting.get(i).is_some_and(|&w| w != 0);
        }
        self.status = s.status;
        self.promisc = s.promisc != 0;
        self.allmulti = s.allmulti != 0;
        // MAC_TABLE_ENTRIES may be different from the saved image.
        let macs = if s.macs.len() > MAC_TABLE_ENTRIES { Vec::new() } else { s.macs.clone() };
        let first_multi = macs.iter().position(|m| m[0] & 1 != 0).unwrap_or(macs.len());
        self.mac_table = MacTable {
            macs,
            first_multi,
            uni_overflow: s.uni_overflow != 0,
            multi_overflow: s.multi_overflow != 0,
        };
        self.vlans.copy_from_slice(&s.vlans);
        self.alluni = s.alluni != 0;
        self.nomulti = s.nomulti != 0;
        self.nouni = s.nouni != 0;
        self.nobcast = s.nobcast != 0;

        self.set_mrg_rx_bufs(s.mergeable_rx_bufs != 0, vdev.has_feature(VIRTIO_F_VERSION_1));
        let offloads = if vdev.has_feature(VIRTIO_NET_F_CTRL_GUEST_OFFLOADS) {
            s.curr_guest_offloads
        } else {
            vdev.guest_features() & GUEST_OFFLOADS_MASK
        };
        self.curr_guest_offloads = offloads;
        // virtio_set_features_nocheck() overwrites them, post_load puts them back.
        self.saved_guest_offloads = Some(offloads);
        Ok(())
    }

    /// `pre_load_queues`: the incoming device has `n` queues.
    pub(super) fn vmstate_pre_load_queues(
        &mut self,
        vdev: &mut VirtIODevice,
        n: usize,
    ) -> Result<()> {
        if n == 3 {
            self.set_multiqueue(vdev, false);
        } else if n == usize::from(self.max_queue_pairs) * 2 + 1 {
            self.set_multiqueue(vdev, true);
        } else {
            return Err(load_error(format!("{n} queues for {} queue pairs", self.max_queue_pairs)));
        }
        Ok(())
    }

    /// `virtio_net_post_load_virtio()`, plus restarting the queues the driver had running.
    pub(super) fn vmstate_post_load(&mut self, vdev: &mut VirtIODevice) -> Result<()> {
        if let Some(offloads) = self.saved_guest_offloads.take() {
            self.curr_guest_offloads = offloads;
            if self.has_vnet_hdr {
                self.apply_guest_offloads();
            }
        }
        // QEMU sends tx_waiting set when its transmit bottom half was pending, with the queue's
        // notifications off, and runs it when the VM starts. Here it means waiting for the
        // peer, so the queue is flushed now and the frames wait in the net layer until then.
        for pair in 0..self.max_queue_pairs {
            self.tx_resume(vdev, pair);
        }
        self.update_queues(vdev, vdev.status());
        Ok(())
    }
}
