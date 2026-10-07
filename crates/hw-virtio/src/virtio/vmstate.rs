// SPDX-License-Identifier: GPL-2.0-or-later

//! The device core's part of migration: what `virtio_save()` writes and `virtio_load()` reads
//! before and after the transport and device model have had their say.
//!
//! [`VirtioVmState`] is plain data. The stream layout, transport hooks included, is written by
//! the machine's VMState code; this side takes a snapshot and puts one back the way
//! `virtio_load()` does: queue registers first, then the features through
//! `virtio_set_features_nocheck()`, then the ring indices from guest memory.

use std::sync::Arc;

use ruvm_base::{Error, Result};

use super::{
    Ring, VIRTIO_F_RING_PACKED, VIRTIO_F_VERSION_1, VIRTIO_NO_VECTOR, VIRTIO_QUEUE_MAX,
    VirtIODevice, VirtioBackend, VirtioDeviceClass, has_feature,
};

/// `VIRTIO_DEVICE_ENDIAN_LITTLE`, the only endianness handled here.
pub const VIRTIO_DEVICE_ENDIAN_LITTLE: u8 = 1;

/// The migrated part of one `VirtQueue`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VirtQueueVmState {
    /// `vring.num`.
    pub num: u32,
    /// `vring.num_default`, not migrated: the largest size the destination accepts.
    pub num_default: u16,
    /// `vring.align`, migrated only by transports with a variable alignment.
    pub align: u32,
    /// `vring.desc`.
    pub desc: u64,
    /// `vring.avail`, in the `virtio/virtqueues` subsection.
    pub avail: u64,
    /// `vring.used`, in the `virtio/virtqueues` subsection.
    pub used: u64,
    /// `last_avail_idx`: the next available entry the device will take.
    pub last_avail_idx: u16,
    /// The MSI-X vector, migrated by the PCI transport.
    pub vector: u16,
    /// `last_avail_wrap_counter`, packed rings only.
    pub last_avail_wrap: bool,
    /// `used_idx`, in the `virtio/packed_virtqueues` subsection.
    pub used_idx: u16,
    /// `used_wrap_counter`, packed rings only.
    pub used_wrap: bool,
    /// `inuse`, packed rings only. Split rings recompute it on load.
    pub inuse: u32,
}

impl Default for VirtQueueVmState {
    fn default() -> Self {
        VirtQueueVmState {
            num: 0,
            num_default: 0,
            align: 0,
            desc: 0,
            avail: 0,
            used: 0,
            last_avail_idx: 0,
            vector: VIRTIO_NO_VECTOR,
            last_avail_wrap: true,
            used_idx: 0,
            used_wrap: true,
            inuse: 0,
        }
    }
}

/// What the device core migrates, the `VirtIODevice` fields of `virtio_save()` and the
/// `virtio` description's subsections.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VirtioVmState {
    /// The virtio device ID, not migrated: it picks the device model's own state.
    pub device_id: u16,
    /// The features the device offers, not migrated: several subsections depend on them.
    pub host_features: u64,
    pub status: u8,
    pub isr: u8,
    pub queue_sel: u16,
    /// The low half goes in the main part of the stream, the whole word in
    /// `virtio/64bit_features`.
    pub guest_features: u64,
    /// `guest_features_ex[1]`, the `virtio/128bit_features` subsection. Always 0 here.
    pub guest_features_hi: u64,
    pub config: Vec<u8>,
    /// Every queue the device has. The stream carries the ones before the first unused one,
    /// [`nvqs`](Self::nvqs) of them.
    pub vqs: Vec<VirtQueueVmState>,
    /// How many queues are in the stream.
    pub nvqs: usize,
    /// `device_endian`, only sent when it is not the default.
    pub device_endian: u8,
    pub broken: bool,
    pub started: bool,
    pub disabled: bool,
    /// The config change MSI-X vector, migrated by the PCI transport.
    pub config_vector: u16,
}

impl VirtioVmState {
    /// `virtio_host_has_feature()`.
    pub fn host_has_feature(&self, bit: u32) -> bool {
        has_feature(self.host_features, bit)
    }

    /// `virtio_vdev_has_feature()`.
    pub fn has_feature(&self, bit: u32) -> bool {
        has_feature(self.guest_features, bit)
    }

    /// Makes room for `n` queues in the stream, as `virtio_load()` does once it has read the
    /// count. Queues past the ones the device has get a maximum size of 0, so loading one fails.
    pub fn resize_queues(&mut self, n: usize) {
        if self.vqs.len() < n {
            self.vqs.resize(n, VirtQueueVmState::default());
        }
        self.nvqs = n;
    }
}

fn load_error(name: &str, msg: impl std::fmt::Display) -> Error {
    Error::generic(format!("{name}: {msg}"))
}

impl VirtioBackend {
    /// The core state as `virtio_save()` sends it. The config space is refreshed from the
    /// device model first, as a driver read would.
    pub fn vmstate_save(&mut self) -> VirtioVmState {
        let mut config = std::mem::take(&mut self.vdev.config);
        self.class.get_config(&self.vdev, &mut config);
        self.vdev.config = config;

        let vdev = &self.vdev;
        let vqs: Vec<_> = vdev
            .vqs
            .iter()
            .map(|q| {
                let mut s = VirtQueueVmState {
                    num: u32::from(q.num),
                    num_default: q.num_default,
                    align: q.align,
                    desc: q.desc,
                    avail: q.avail,
                    used: q.used,
                    vector: q.vector,
                    inuse: q.inuse,
                    ..VirtQueueVmState::default()
                };
                match &q.ring {
                    Some(Ring::Split(r)) => {
                        s.last_avail_idx = r.next_avail();
                        s.used_idx = r.next_used();
                    }
                    Some(Ring::Packed(r)) => {
                        s.last_avail_idx = r.next_avail();
                        s.last_avail_wrap = r.avail_wrap_counter();
                        s.used_idx = r.next_used();
                        s.used_wrap = r.used_wrap_counter();
                    }
                    None => {}
                }
                s
            })
            .collect();
        let nvqs = vqs.iter().position(|q| q.num == 0).unwrap_or(vqs.len());
        VirtioVmState {
            device_id: vdev.device_id,
            host_features: vdev.host_features,
            status: vdev.status,
            isr: vdev.isr,
            queue_sel: vdev.queue_sel,
            guest_features: vdev.guest_features,
            guest_features_hi: 0,
            config: vdev.config.clone(),
            vqs,
            nvqs,
            device_endian: VIRTIO_DEVICE_ENDIAN_LITTLE,
            broken: vdev.broken,
            started: vdev.started,
            disabled: vdev.disabled,
            config_vector: vdev.config_vector,
        }
    }

    /// `virtio_load()` from the queue registers on, then `device`, which loads the device
    /// model's own state (`vdc->load` and `vdc->vmsd`), then the model's `post_load` hook.
    ///
    /// The transport's registers must be back before this runs, and the transport recomputes
    /// its interrupt line afterwards. Guest memory must be loaded, since the used and available
    /// indices are read from it.
    pub fn vmstate_load(
        &mut self,
        s: &VirtioVmState,
        device: impl FnOnce(&mut VirtIODevice, &mut dyn VirtioDeviceClass) -> Result<()>,
    ) -> Result<()> {
        let name = self.vdev.name.clone();
        if usize::from(s.queue_sel) >= VIRTIO_QUEUE_MAX {
            return Err(load_error(&name, format!("bad queue_sel {}", s.queue_sel)));
        }
        if s.device_endian != VIRTIO_DEVICE_ENDIAN_LITTLE {
            return Err(load_error(
                &name,
                format!("unsupported device endian {}", s.device_endian),
            ));
        }
        if s.guest_features_hi != 0 {
            return Err(load_error(
                &name,
                format!("Features 0x{:x} above bit 63 unsupported", s.guest_features_hi),
            ));
        }
        let nvqs = s.nvqs.min(s.vqs.len());
        if nvqs > VIRTIO_QUEUE_MAX {
            return Err(load_error(&name, format!("Invalid number of virtqueues: {nvqs:#x}")));
        }
        self.class.pre_load_queues(&mut self.vdev, nvqs)?;

        let vdev = &mut self.vdev;
        vdev.status = s.status;
        vdev.isr = s.isr;
        vdev.queue_sel = s.queue_sel;
        // A config space of another size: take what fits, drop the rest.
        let n = s.config.len().min(vdev.config.len());
        vdev.config[..n].copy_from_slice(&s.config[..n]);

        for (i, q) in s.vqs[..nvqs].iter().enumerate() {
            let max = vdev.vqs.get(i).map_or(0, |vq| vq.num_default);
            let num = match u16::try_from(q.num) {
                Ok(num) if num <= max && i < vdev.vqs.len() => num,
                _ => {
                    return Err(load_error(
                        &name,
                        format!("VQ {i} vring.num {} exceeds allocated max {max}", q.num),
                    ));
                }
            };
            if q.desc == 0 && q.last_avail_idx != 0 {
                return Err(load_error(
                    &name,
                    format!(
                        "VQ {i} address 0x0 inconsistent with Host index {:#x}",
                        q.last_avail_idx
                    ),
                ));
            }
            let vq = &mut vdev.vqs[i];
            vq.num = num;
            vq.align = q.align;
            vq.desc = q.desc;
            vq.avail = q.avail;
            vq.used = q.used;
            vq.vector = q.vector;
            vq.notification = true;
            vq.inuse = 0;
            vq.ring = None;
            vq.ring_error = None;
        }
        vdev.config_vector = s.config_vector;
        vdev.notify_vector(VIRTIO_NO_VECTOR);
        // The whole feature word is in place before the device model loads, as the
        // `virtio/64bit_features` subsection puts it there in QEMU. Device models compare
        // against it in `set_features`, so nothing they loaded is reset below.
        vdev.guest_features = s.guest_features & vdev.host_features;

        let class = &mut *self.class;
        device(&mut self.vdev, class)?;

        let vdev = &mut self.vdev;
        vdev.broken = s.broken;
        vdev.started = s.started;
        vdev.disabled = s.disabled;

        // virtio_set_features_nocheck(), which also builds the rings from the registers.
        if let Err(e) = self.set_features_nocheck(s.guest_features) {
            return Err(load_error(
                &name,
                format!(
                    "Features {:#x} unsupported. Allowed features: {:#x}: {e}",
                    s.guest_features, self.vdev.host_features
                ),
            ));
        }
        let vdev = &mut self.vdev;
        if !vdev.started && !vdev.has_feature(VIRTIO_F_VERSION_1) {
            vdev.start_on_kick = true;
        }

        let features = vdev.guest_features;
        let legacy = !has_feature(features, VIRTIO_F_VERSION_1);
        let packed = has_feature(features, VIRTIO_F_RING_PACKED);
        let mem = Arc::clone(&vdev.mem);
        let mut errors = Vec::new();
        for (i, q) in s.vqs[..nvqs].iter().enumerate() {
            if q.desc == 0 {
                continue;
            }
            let vq = &mut vdev.vqs[i];
            if legacy {
                vq.update_rings(features);
            } else {
                vq.rebuild(features);
            }
            if packed {
                let bad = match vq.ring.as_mut() {
                    Some(Ring::Packed(r)) => {
                        r.set_avail_state(q.last_avail_idx, q.last_avail_wrap).is_err()
                            || r.set_used_state(q.used_idx, q.used_wrap).is_err()
                    }
                    _ => false,
                };
                if bad {
                    return Err(load_error(&name, format!("VQ {i} bad packed ring state")));
                }
                vq.inuse = q.inuse;
                continue;
            }
            let num = vq.num;
            let (avail_addr, used_addr) = (vq.avail, vq.used);
            let Some(Ring::Split(ring)) = vq.ring.as_mut() else {
                continue;
            };
            let avail_idx = mem.read_u16(avail_addr + 2).unwrap_or(0);
            let nheads = avail_idx.wrapping_sub(q.last_avail_idx);
            if nheads > num {
                ring.set_next_avail(q.last_avail_idx);
                ring.set_next_used(0);
                vq.inuse = 0;
                errors.push(format!(
                    "VQ {i} size {num:#x} Guest index {avail_idx:#x} inconsistent with Host \
                     index {:#x}: delta {nheads:#x}",
                    q.last_avail_idx
                ));
                continue;
            }
            let used_idx = mem.read_u16(used_addr + 2).unwrap_or(0);
            let inuse = q.last_avail_idx.wrapping_sub(used_idx);
            if inuse > num {
                return Err(load_error(
                    &name,
                    format!(
                        "VQ {i} size {num:#x} < last_avail_idx {:#x} - used_idx {used_idx:#x}",
                        q.last_avail_idx
                    ),
                ));
            }
            ring.set_next_avail(q.last_avail_idx);
            ring.set_next_used(used_idx);
            vq.inuse = u32::from(inuse);
        }
        for e in errors {
            vdev.error(&e);
        }

        self.class.post_load(&mut self.vdev)
    }
}

impl VirtIODevice {
    /// `virtqueue_rewind()`: hands the last `num` chains taken from queue `n` back, so the
    /// next pop returns them again. Used for chains a device held on to and that were not
    /// migrated. Split rings only; returns whether it happened.
    pub fn rewind(&mut self, n: u16, num: u32) -> bool {
        let Some(q) = self.vqs.get_mut(usize::from(n)) else {
            return false;
        };
        if num > q.inuse {
            return false;
        }
        let Some(Ring::Split(ring)) = q.ring.as_mut() else {
            return false;
        };
        ring.set_next_avail(ring.next_avail().wrapping_sub(num as u16));
        q.inuse -= num;
        true
    }
}
