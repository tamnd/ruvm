// SPDX-License-Identifier: GPL-2.0-or-later

//! What the virtio-pci transport adds to a virtio device's migration stream: the PCI function
//! (`save_config`), the MSI-X vectors of the queues (`save_queue`) and the `virtio_pci` extra
//! state with the modern queue registers.

use ruvm_base::{Error, Result};
use ruvm_hw_pci::PciDeviceVmState;
use ruvm_hw_pci::regs::{PCI_STATUS, PCI_STATUS_INTERRUPT};

use super::{PciQueue, VirtioPci};
use crate::virtio::{
    VIRTIO_NO_VECTOR, VIRTIO_QUEUE_MAX, VirtIODevice, VirtioDeviceClass, VirtioVmState,
};

/// `VirtIOPCIQueue` as `virtio_pci/modern_queue_state` has it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VirtioPciQueueVmState {
    pub num: u16,
    pub enabled: bool,
    pub desc: [u32; 2],
    pub avail: [u32; 2],
    pub used: [u32; 2],
}

/// A virtio PCI device's migration state: transport and device core.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VirtioPciVmState {
    /// The `PCIDevice` section `pci_device_save()` writes, with the interrupt status bit
    /// cleared as `virtio_pci_save_config()` does.
    pub pci: PciDeviceVmState,
    /// How many MSI-X vectors the function has, 0 without MSI-X. Not migrated, it sizes
    /// [`msix_table`](Self::msix_table) and [`msix_pba`](Self::msix_pba) and says whether the
    /// vectors are in the stream.
    pub msix_vectors: u32,
    /// `msix_save()`: the vector table.
    pub msix_table: Vec<u8>,
    /// `msix_save()`: the pending bits.
    pub msix_pba: Vec<u8>,
    /// Whether the modern interface is offered, which is when `virtio_pci/modern_state` is
    /// sent. Not migrated.
    pub modern: bool,
    pub dfselect: u32,
    pub gfselect: u32,
    /// `guest_features[4]`. Words 2 and 3 go in `virtio_pci/modern_state/features128` and
    /// must be 0 here.
    pub guest_features: [u32; 4],
    /// [`VIRTIO_QUEUE_MAX`] entries.
    pub vqs: Vec<VirtioPciQueueVmState>,
    /// The device core.
    pub vdev: VirtioVmState,
}

impl VirtioPciVmState {
    /// Whether the MSI-X state and vectors are in the stream, `msix_present()`.
    pub fn msix_present(&self) -> bool {
        self.msix_vectors != 0
    }
}

fn load_error(name: &str, msg: impl std::fmt::Display) -> Error {
    Error::generic(format!("{name}: {msg}"))
}

impl VirtioPci {
    /// The state `virtio_save()` sends for this function. `None` if called from inside the
    /// device's own processing.
    pub fn vmstate_save(&self) -> Option<VirtioPciVmState> {
        let mut st = self.inner.lock()?;
        let mut pci = self.pci.vmstate_save();
        pci.config[PCI_STATUS] &= !(PCI_STATUS_INTERRUPT as u8);
        let msix_vectors =
            if self.pci.msix_present() { self.pci.msix_nr_vectors_allocated() } else { 0 };
        let (msix_table, msix_pba) = self.pci.msix_vmstate_save();
        let mut vqs = vec![VirtioPciQueueVmState::default(); VIRTIO_QUEUE_MAX];
        for (q, s) in st.vqs.iter().zip(vqs.iter_mut()) {
            *s = VirtioPciQueueVmState {
                num: q.num,
                enabled: q.enabled,
                desc: q.desc,
                avail: q.avail,
                used: q.used,
            };
        }
        let gf = st.guest_features;
        Some(VirtioPciVmState {
            pci,
            msix_vectors,
            msix_table,
            msix_pba,
            modern: self.inner.modern,
            dfselect: st.dfselect,
            gfselect: st.gfselect,
            guest_features: [gf[0], gf[1], 0, 0],
            vqs,
            vdev: st.backend.vmstate_save(),
        })
    }

    /// `virtio_load()` for a PCI function: config space and MSI-X (`virtio_pci_load_config()`),
    /// the queue vectors (`virtio_pci_load_queue()`), the modern registers, then the device
    /// core and `device`, which loads the device model's own state.
    pub fn vmstate_load(
        &self,
        s: &VirtioPciVmState,
        device: impl FnOnce(&mut VirtIODevice, &mut dyn VirtioDeviceClass) -> Result<()>,
    ) -> Result<()> {
        let name = self.pci.name().to_string();
        let mut st =
            self.inner.lock().ok_or_else(|| load_error(&name, "loaded from inside the device"))?;
        if s.guest_features[2] != 0 || s.guest_features[3] != 0 {
            return Err(load_error(&name, "extended guest features are not supported"));
        }
        self.pci.vmstate_load(&s.pci).map_err(Error::generic)?;
        let msix = self.pci.msix_present();
        if msix != s.msix_present() {
            return Err(load_error(&name, "MSI-X presence differs"));
        }
        self.pci.msix_unuse_all_vectors();
        self.pci.msix_vmstate_load(&s.msix_table, &s.msix_pba).map_err(Error::generic)?;

        let nvectors = self.inner.nvectors();
        let use_vector = |v: u16| -> Result<u16> {
            if !msix || v == VIRTIO_NO_VECTOR {
                return Ok(VIRTIO_NO_VECTOR);
            }
            if u32::from(v) >= nvectors || u32::from(v) >= self.pci.msix_nr_vectors_allocated() {
                return Err(load_error(&name, format!("MSI-X vector {v} out of range")));
            }
            self.pci.msix_vector_use(u32::from(v));
            Ok(v)
        };
        let mut vdev = s.vdev.clone();
        vdev.config_vector = use_vector(s.vdev.config_vector)?;
        let nvqs = vdev.nvqs.min(vdev.vqs.len());
        for q in &mut vdev.vqs[..nvqs] {
            q.vector = use_vector(q.vector)?;
        }

        st.dfselect = s.dfselect;
        st.gfselect = s.gfselect;
        st.guest_features = [s.guest_features[0], s.guest_features[1]];
        for (q, v) in st.vqs.iter_mut().zip(s.vqs.iter()) {
            *q = PciQueue {
                num: v.num,
                enabled: v.enabled,
                reset: false,
                desc: v.desc,
                avail: v.avail,
                used: v.used,
            };
        }
        st.backend.vmstate_load(&vdev, device)
    }
}
