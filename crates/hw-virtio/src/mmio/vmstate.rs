// SPDX-License-Identifier: GPL-2.0-or-later

//! What the virtio-mmio transport adds to a virtio device's migration stream: the three
//! selector registers (`save_config`) and, for a modern transport, the `virtio_mmio` extra
//! state with the queue registers.

use ruvm_base::{Error, Result};

use super::{MmioQueue, VirtioMmio};
use crate::virtio::{VIRTIO_QUEUE_MAX, VirtIODevice, VirtioDeviceClass, VirtioVmState};

/// `VirtIOMMIOQueue` as `virtio_mmio/queue_state` has it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VirtioMmioQueueVmState {
    pub num: u16,
    pub enabled: bool,
    pub desc: [u32; 2],
    pub avail: [u32; 2],
    pub used: [u32; 2],
}

/// A virtio-mmio device's migration state: transport and device core.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VirtioMmioVmState {
    pub host_features_sel: u32,
    pub guest_features_sel: u32,
    pub guest_page_shift: u32,
    /// Whether the transport is legacy, in which case the `virtio_mmio` extra state is not
    /// sent. Not migrated.
    pub legacy: bool,
    pub guest_features: [u32; 2],
    /// [`VIRTIO_QUEUE_MAX`] entries.
    pub vqs: Vec<VirtioMmioQueueVmState>,
    /// The device core.
    pub vdev: VirtioVmState,
}

fn load_error(msg: impl std::fmt::Display) -> Error {
    Error::generic(format!("virtio-mmio: {msg}"))
}

impl VirtioMmio {
    /// The state `virtio_save()` sends for this transport. `None` if no device is plugged in,
    /// in which case there is no section, or if called from inside the device's processing.
    pub fn vmstate_save(&self) -> Option<VirtioMmioVmState> {
        let mut st = self.lock()?;
        let st = &mut *st;
        let vdev = st.backend.as_mut()?.vmstate_save();
        let mut vqs = vec![VirtioMmioQueueVmState::default(); VIRTIO_QUEUE_MAX];
        for (q, s) in st.vqs.iter().zip(vqs.iter_mut()) {
            *s = VirtioMmioQueueVmState {
                num: q.num,
                enabled: q.enabled,
                desc: q.desc,
                avail: q.avail,
                used: q.used,
            };
        }
        Some(VirtioMmioVmState {
            host_features_sel: st.host_features_sel,
            guest_features_sel: st.guest_features_sel,
            guest_page_shift: st.guest_page_shift,
            legacy: self.legacy,
            guest_features: st.guest_features,
            vqs,
            vdev,
        })
    }

    /// `virtio_load()` for this transport: the registers, then the device core and `device`,
    /// which loads the device model's own state.
    pub fn vmstate_load(
        &self,
        s: &VirtioMmioVmState,
        device: impl FnOnce(&mut VirtIODevice, &mut dyn VirtioDeviceClass) -> Result<()>,
    ) -> Result<()> {
        let mut st = self.lock().ok_or_else(|| load_error("loaded from inside the device"))?;
        let st = &mut *st;
        let Some(backend) = st.backend.as_mut() else {
            return Err(load_error("no device plugged in"));
        };
        st.host_features_sel = s.host_features_sel;
        st.guest_features_sel = s.guest_features_sel;
        st.guest_page_shift = s.guest_page_shift;
        if !self.legacy {
            st.guest_features = s.guest_features;
            for (q, v) in st.vqs.iter_mut().zip(s.vqs.iter()) {
                *q = MmioQueue {
                    num: v.num,
                    enabled: v.enabled,
                    desc: v.desc,
                    avail: v.avail,
                    used: v.used,
                };
            }
        }
        backend.vmstate_load(&s.vdev, device)
    }
}
