// SPDX-License-Identifier: GPL-2.0-or-later

//! The virtio device sections: `VMSTATE_VIRTIO_DEVICE` with `virtio_save()` and `virtio_load()`
//! from hw/virtio/virtio.c, the virtio-pci (hw/virtio/virtio-pci.c) and virtio-mmio
//! (hw/virtio/virtio-mmio.c) transport hooks, and the device models' own parts: virtio-blk,
//! virtio-net, virtio-scsi with its `scsi-disk` devices, virtio-rng, virtio-balloon and
//! virtio-serial with the console port.
//!
//! A device section has one field whose info writes the whole of `virtio_save()`: the
//! transport's config, the core registers, the queues, the model's state and then the `virtio`
//! description with its subsections. The state is [`VirtioSection`], plain data the transports
//! produce and take back.

use std::sync::{Arc, LazyLock};

use ruvm_base::{Result, bail, err};
use ruvm_hw_storage::scsi::{SCSI_SENSE_BUF_SIZE, SCSI_SENSE_BUF_SIZE_OLD, ScsiDiskVmState};
use ruvm_hw_virtio::balloon::VIRTIO_ID_BALLOON;
use ruvm_hw_virtio::blk::VIRTIO_ID_BLOCK;
use ruvm_hw_virtio::console::VIRTIO_ID_CONSOLE;
use ruvm_hw_virtio::net::VIRTIO_ID_NET;
use ruvm_hw_virtio::rng::VIRTIO_ID_RNG;
use ruvm_hw_virtio::scsi::VIRTIO_ID_SCSI;
use ruvm_hw_virtio::virtio::{
    VIRTIO_DEVICE_ENDIAN_LITTLE, VIRTIO_F_VERSION_1, VIRTIO_NO_VECTOR, VIRTIO_QUEUE_MAX,
};
use ruvm_hw_virtio::{
    VirtIODevice, VirtQueueVmState, VirtioBackend, VirtioBalloon, VirtioBalloonVmState,
    VirtioConsole, VirtioConsolePortVmState, VirtioConsoleVmState, VirtioDeviceClass, VirtioMmio,
    VirtioMmioQueueVmState, VirtioMmioVmState, VirtioNet, VirtioNetVmState, VirtioPci,
    VirtioPciQueueVmState, VirtioPciVmState, VirtioScsi, VirtioVmState,
};
use ruvm_migration::SaveVm;
use ruvm_vmstate::info::Uint16Equal;
use ruvm_vmstate::{
    StreamReader, StreamWriter, VmStateDescription, VmStateField, VmStateInfo, vmstate_load_state,
    vmstate_save_state,
};

use super::pci::VMSTATE_PCI_DEVICE;

/// `VIRTIO_F_RING_PACKED`.
const VIRTIO_F_RING_PACKED: u32 = 34;
/// `VIRTIO_NET_F_CTRL_GUEST_OFFLOADS`.
const VIRTIO_NET_F_CTRL_GUEST_OFFLOADS: u32 = 2;
/// `MAC_TABLE_ENTRIES`.
const MAC_TABLE_ENTRIES: usize = 64;
/// `MAX_VLAN >> 3`: the bytes of the VLAN filter bitmap.
const VLAN_BYTES: usize = 4096 >> 3;
/// `VIRTIO_NET_VM_VERSION`.
const VIRTIO_NET_VM_VERSION: i32 = 11;

/// A device model and its own part of the stream.
#[derive(Clone, Debug)]
pub(crate) enum VirtioModel {
    /// virtio-blk: the request list, always empty here.
    Blk,
    /// virtio-rng: nothing of its own.
    Rng,
    /// virtio-scsi: nothing of its own, the disks have their sections.
    Scsi,
    /// virtio-net: `virtio-net-device`.
    Net(NetDevice),
    /// virtio-balloon: `virtio-balloon-device`.
    Balloon(VirtioBalloonVmState),
    /// virtio-serial: what `virtio_serial_save_device()` writes.
    Console(VirtioConsoleVmState),
}

/// `virtio-net-device` with the driver feature its last field depends on.
#[derive(Clone, Debug, Default)]
pub(crate) struct NetDevice {
    s: VirtioNetVmState,
    /// Whether the driver has `VIRTIO_NET_F_CTRL_GUEST_OFFLOADS`, from the low feature word.
    ctrl_guest_offloads: bool,
}

/// The transport's part.
#[derive(Clone, Debug)]
pub(crate) enum Transport {
    Pci(Box<VirtioPciVmState>),
    Mmio(Box<VirtioMmioVmState>),
}

/// One virtio device section.
#[derive(Clone, Debug)]
pub(crate) struct VirtioSection {
    transport: Transport,
    model: VirtioModel,
}

impl VirtioSection {
    fn vdev(&mut self) -> &mut VirtioVmState {
        match &mut self.transport {
            Transport::Pci(p) => &mut p.vdev,
            Transport::Mmio(m) => &mut m.vdev,
        }
    }

    fn vdev_ref(&self) -> &VirtioVmState {
        match &self.transport {
            Transport::Pci(p) => &p.vdev,
            Transport::Mmio(m) => &m.vdev,
        }
    }

    /// `has_variable_vring_alignment`, which only virtio-mmio has.
    fn mmio(&self) -> bool {
        matches!(self.transport, Transport::Mmio(_))
    }

    /// Whether the queue and config vectors are in the stream.
    fn msix(&self) -> bool {
        match &self.transport {
            Transport::Pci(p) => p.msix_present(),
            Transport::Mmio(_) => false,
        }
    }

    /// Gives every array QEMU sends in full its [`VIRTIO_QUEUE_MAX`] entries.
    fn pad(&mut self) {
        match &mut self.transport {
            Transport::Pci(p) => p.vqs.resize(VIRTIO_QUEUE_MAX, VirtioPciQueueVmState::default()),
            Transport::Mmio(m) => m.vqs.resize(VIRTIO_QUEUE_MAX, VirtioMmioQueueVmState::default()),
        }
        let vdev = self.vdev();
        if vdev.vqs.len() < VIRTIO_QUEUE_MAX {
            vdev.vqs.resize(VIRTIO_QUEUE_MAX, VirtQueueVmState::default());
        }
    }
}

fn has_feature(features: u64, bit: u32) -> bool {
    features & (1 << bit) != 0
}

// The `virtio` description and its subsections.

/// `vmstate_virtqueue`.
static VMSTATE_VIRTQUEUE: LazyLock<VmStateDescription<VirtQueueVmState>> = LazyLock::new(|| {
    type Q = VirtQueueVmState;
    VmStateDescription::new("virtqueue_state").version_id(1).minimum_version_id(1).fields([
        VmStateField::scalar("vring.avail", |q: &mut Q| &mut q.avail),
        VmStateField::scalar("vring.used", |q: &mut Q| &mut q.used),
    ])
});

/// `vmstate_packed_virtqueue`.
static VMSTATE_PACKED_VIRTQUEUE: LazyLock<VmStateDescription<VirtQueueVmState>> =
    LazyLock::new(|| {
        type Q = VirtQueueVmState;
        VmStateDescription::new("packed_virtqueue_state")
            .version_id(1)
            .minimum_version_id(1)
            .fields([
                VmStateField::scalar("last_avail_idx", |q: &mut Q| &mut q.last_avail_idx),
                VmStateField::scalar("last_avail_wrap_counter", |q: &mut Q| &mut q.last_avail_wrap),
                VmStateField::scalar("used_idx", |q: &mut Q| &mut q.used_idx),
                VmStateField::scalar("used_wrap_counter", |q: &mut Q| &mut q.used_wrap),
                VmStateField::scalar("inuse", |q: &mut Q| &mut q.inuse),
            ])
    });

/// `vmstate_ringsize`: an unused word per queue.
static VMSTATE_RINGSIZE: LazyLock<VmStateDescription<VirtQueueVmState>> = LazyLock::new(|| {
    VmStateDescription::new("ringsize_state")
        .version_id(1)
        .minimum_version_id(1)
        .field(VmStateField::unused(4))
});

type S = VirtioSection;

/// `vmstate_virtio_device_endian`. Only little endian devices are handled, so it is never
/// sent and an incoming one must say little endian.
static VMSTATE_VIRTIO_DEVICE_ENDIAN: LazyLock<VmStateDescription<S>> = LazyLock::new(|| {
    VmStateDescription::new("virtio/device_endian")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|s: &S| s.vdev_ref().device_endian != VIRTIO_DEVICE_ENDIAN_LITTLE)
        .field(VmStateField::scalar("device_endian", |s: &mut S| &mut s.vdev().device_endian))
});

/// `vmstate_virtio_128bit_features`.
static VMSTATE_VIRTIO_128BIT_FEATURES: LazyLock<VmStateDescription<S>> = LazyLock::new(|| {
    VmStateDescription::new("virtio/128bit_features")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|s: &S| s.vdev_ref().guest_features_hi != 0)
        .field(VmStateField::scalar("guest_features_ex[1]", |s: &mut S| {
            &mut s.vdev().guest_features_hi
        }))
});

/// `vmstate_virtio_64bit_features`.
static VMSTATE_VIRTIO_64BIT_FEATURES: LazyLock<VmStateDescription<S>> = LazyLock::new(|| {
    VmStateDescription::new("virtio/64bit_features")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|s: &S| s.vdev_ref().host_features >> 32 != 0)
        .field(VmStateField::scalar("guest_features", |s: &mut S| &mut s.vdev().guest_features))
});

/// `vmstate_virtio_virtqueues`: the available and used ring addresses of every queue.
static VMSTATE_VIRTIO_VIRTQUEUES: LazyLock<VmStateDescription<S>> = LazyLock::new(|| {
    VmStateDescription::new("virtio/virtqueues")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|s: &S| s.vdev_ref().host_has_feature(VIRTIO_F_VERSION_1))
        .field(VmStateField::struct_varray(
            "vq",
            |_: &S| VIRTIO_QUEUE_MAX,
            &VMSTATE_VIRTQUEUE,
            |s: &mut S| &mut s.vdev().vqs,
        ))
});

/// `vmstate_virtio_ringsize`, never sent.
static VMSTATE_VIRTIO_RINGSIZE: LazyLock<VmStateDescription<S>> = LazyLock::new(|| {
    VmStateDescription::new("virtio/ringsize")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|_: &S| false)
        .field(VmStateField::struct_varray(
            "vq",
            |_: &S| VIRTIO_QUEUE_MAX,
            &VMSTATE_RINGSIZE,
            |s: &mut S| &mut s.vdev().vqs,
        ))
});

/// `vmstate_virtio_broken`.
static VMSTATE_VIRTIO_BROKEN: LazyLock<VmStateDescription<S>> = LazyLock::new(|| {
    VmStateDescription::new("virtio/broken")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|s: &S| s.vdev_ref().broken)
        .field(VmStateField::scalar("broken", |s: &mut S| &mut s.vdev().broken))
});

/// `vmstate_virtio_started`.
static VMSTATE_VIRTIO_STARTED: LazyLock<VmStateDescription<S>> = LazyLock::new(|| {
    VmStateDescription::new("virtio/started")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|s: &S| s.vdev_ref().started)
        .field(VmStateField::scalar("started", |s: &mut S| &mut s.vdev().started))
});

/// `vmstate_virtio_packed_virtqueues`.
static VMSTATE_VIRTIO_PACKED_VIRTQUEUES: LazyLock<VmStateDescription<S>> = LazyLock::new(|| {
    VmStateDescription::new("virtio/packed_virtqueues")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|s: &S| s.vdev_ref().host_has_feature(VIRTIO_F_RING_PACKED))
        .field(VmStateField::struct_varray(
            "vq",
            |_: &S| VIRTIO_QUEUE_MAX,
            &VMSTATE_PACKED_VIRTQUEUE,
            |s: &mut S| &mut s.vdev().vqs,
        ))
});

/// `vmstate_virtio_disabled`.
static VMSTATE_VIRTIO_DISABLED: LazyLock<VmStateDescription<S>> = LazyLock::new(|| {
    VmStateDescription::new("virtio/disabled")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|s: &S| s.vdev_ref().disabled)
        .field(VmStateField::scalar("disabled", |s: &mut S| &mut s.vdev().disabled))
});

/// `vmstate_info_extra_state`: the transport's own description.
#[derive(Debug)]
struct ExtraState;

impl VmStateInfo<Transport> for ExtraState {
    fn name(&self) -> &'static str {
        "virtqueue_extra_state"
    }

    fn load(&self, f: &mut StreamReader<'_>, v: &mut Transport, _size: usize) -> Result<()> {
        match v {
            Transport::Pci(p) => vmstate_load_state(f, &VMSTATE_VIRTIO_PCI, p, 1),
            Transport::Mmio(m) => vmstate_load_state(f, &VMSTATE_VIRTIO_MMIO, m, 1),
        }
    }

    fn save(&self, f: &mut StreamWriter, v: &Transport, _size: usize) -> Result<()> {
        match v.clone() {
            Transport::Pci(mut p) => vmstate_save_state(f, &VMSTATE_VIRTIO_PCI, &mut p),
            Transport::Mmio(mut m) => vmstate_save_state(f, &VMSTATE_VIRTIO_MMIO, &mut m),
        }
    }
}

/// `vmstate_virtio_extra_state`: `has_extra_state()` is always true on PCI and true for a
/// modern virtio-mmio transport.
static VMSTATE_VIRTIO_EXTRA_STATE: LazyLock<VmStateDescription<S>> = LazyLock::new(|| {
    VmStateDescription::new("virtio/extra_state")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|s: &S| match &s.transport {
            Transport::Pci(_) => true,
            Transport::Mmio(m) => !m.legacy,
        })
        .field(VmStateField::single("extra_state", &ExtraState, |s: &mut S| &mut s.transport))
});

/// `vmstate_virtio`: only subsections, after everything else.
static VMSTATE_VIRTIO: LazyLock<VmStateDescription<S>> = LazyLock::new(|| {
    VmStateDescription::new("virtio")
        .version_id(1)
        .minimum_version_id(1)
        .subsection(&VMSTATE_VIRTIO_DEVICE_ENDIAN)
        .subsection(&VMSTATE_VIRTIO_128BIT_FEATURES)
        .subsection(&VMSTATE_VIRTIO_64BIT_FEATURES)
        .subsection(&VMSTATE_VIRTIO_VIRTQUEUES)
        .subsection(&VMSTATE_VIRTIO_RINGSIZE)
        .subsection(&VMSTATE_VIRTIO_BROKEN)
        .subsection(&VMSTATE_VIRTIO_EXTRA_STATE)
        .subsection(&VMSTATE_VIRTIO_STARTED)
        .subsection(&VMSTATE_VIRTIO_PACKED_VIRTQUEUES)
        .subsection(&VMSTATE_VIRTIO_DISABLED)
});

// virtio-pci.

/// `vmstate_virtio_pci_modern_queue_state`.
static VMSTATE_VIRTIO_PCI_MODERN_QUEUE: LazyLock<VmStateDescription<VirtioPciQueueVmState>> =
    LazyLock::new(|| {
        type Q = VirtioPciQueueVmState;
        VmStateDescription::new("virtio_pci/modern_queue_state")
            .version_id(1)
            .minimum_version_id(1)
            .fields([
                VmStateField::scalar("num", |q: &mut Q| &mut q.num),
                // `enabled` was stored as be16.
                VmStateField::unused(1),
                VmStateField::scalar("enabled", |q: &mut Q| &mut q.enabled),
                VmStateField::array("desc", |q: &mut Q| &mut q.desc),
                VmStateField::array("avail", |q: &mut Q| &mut q.avail),
                VmStateField::array("used", |q: &mut Q| &mut q.used),
            ])
    });

/// `vmstate_virtio_pci_modern_state_features128`.
static VMSTATE_VIRTIO_PCI_FEATURES128: LazyLock<VmStateDescription<VirtioPciVmState>> =
    LazyLock::new(|| {
        type P = VirtioPciVmState;
        VmStateDescription::new("virtio_pci/modern_state/features128")
            .version_id(1)
            .minimum_version_id(1)
            .needed(|p: &P| p.guest_features[2] | p.guest_features[3] != 0)
            .field(VmStateField::varray(
                "guest_features",
                |_: &P| 2,
                |p: &mut P| &mut p.guest_features[2..],
            ))
    });

/// `vmstate_virtio_pci_modern_state_sub`.
static VMSTATE_VIRTIO_PCI_MODERN_STATE: LazyLock<VmStateDescription<VirtioPciVmState>> =
    LazyLock::new(|| {
        type P = VirtioPciVmState;
        VmStateDescription::new("virtio_pci/modern_state")
            .version_id(1)
            .minimum_version_id(1)
            .needed(|p: &P| p.modern)
            .fields([
                VmStateField::scalar("dfselect", |p: &mut P| &mut p.dfselect),
                VmStateField::scalar("gfselect", |p: &mut P| &mut p.gfselect),
                VmStateField::varray(
                    "guest_features",
                    |_: &P| 2,
                    |p: &mut P| &mut p.guest_features[..2],
                ),
                VmStateField::struct_varray(
                    "vqs",
                    |_: &P| VIRTIO_QUEUE_MAX,
                    &VMSTATE_VIRTIO_PCI_MODERN_QUEUE,
                    |p: &mut P| &mut p.vqs,
                ),
            ])
            .subsection(&VMSTATE_VIRTIO_PCI_FEATURES128)
    });

/// `vmstate_virtio_pci`.
static VMSTATE_VIRTIO_PCI: LazyLock<VmStateDescription<VirtioPciVmState>> = LazyLock::new(|| {
    VmStateDescription::new("virtio_pci")
        .version_id(1)
        .minimum_version_id(1)
        .subsection(&VMSTATE_VIRTIO_PCI_MODERN_STATE)
});

// virtio-mmio.

/// `vmstate_virtio_mmio_queue_state`.
static VMSTATE_VIRTIO_MMIO_QUEUE: LazyLock<VmStateDescription<VirtioMmioQueueVmState>> =
    LazyLock::new(|| {
        type Q = VirtioMmioQueueVmState;
        VmStateDescription::new("virtio_mmio/queue_state")
            .version_id(1)
            .minimum_version_id(1)
            .fields([
                VmStateField::scalar("num", |q: &mut Q| &mut q.num),
                VmStateField::scalar("enabled", |q: &mut Q| &mut q.enabled),
                VmStateField::array("desc", |q: &mut Q| &mut q.desc),
                VmStateField::array("avail", |q: &mut Q| &mut q.avail),
                VmStateField::array("used", |q: &mut Q| &mut q.used),
            ])
    });

/// `vmstate_virtio_mmio_state_sub`, always sent.
static VMSTATE_VIRTIO_MMIO_STATE: LazyLock<VmStateDescription<VirtioMmioVmState>> =
    LazyLock::new(|| {
        type M = VirtioMmioVmState;
        VmStateDescription::new("virtio_mmio/state").version_id(1).minimum_version_id(1).fields([
            VmStateField::array("guest_features", |m: &mut M| &mut m.guest_features),
            VmStateField::struct_varray(
                "vqs",
                |_: &M| VIRTIO_QUEUE_MAX,
                &VMSTATE_VIRTIO_MMIO_QUEUE,
                |m: &mut M| &mut m.vqs,
            ),
        ])
    });

/// `vmstate_virtio_mmio`.
static VMSTATE_VIRTIO_MMIO: LazyLock<VmStateDescription<VirtioMmioVmState>> = LazyLock::new(|| {
    VmStateDescription::new("virtio_mmio")
        .version_id(1)
        .minimum_version_id(1)
        .subsection(&VMSTATE_VIRTIO_MMIO_STATE)
});

// virtio-net.

/// The first queue pair's `tx_waiting`, `VMSTATE_STRUCT_POINTER(vqs, ...)`.
#[derive(Debug)]
struct TxWaitingFirst;

impl VmStateInfo<Vec<u32>> for TxWaitingFirst {
    fn name(&self) -> &'static str {
        "virtio-net-queue-tx_waiting"
    }

    fn load(&self, f: &mut StreamReader<'_>, v: &mut Vec<u32>, _size: usize) -> Result<()> {
        let w = f.get_be32();
        match v.first_mut() {
            Some(x) => *x = w,
            None => v.push(w),
        }
        Ok(())
    }

    fn save(&self, f: &mut StreamWriter, v: &Vec<u32>, _size: usize) -> Result<()> {
        f.put_be32(v.first().copied().unwrap_or(0));
        Ok(())
    }
}

/// `vmstate_virtio_net_tx_waiting`: the other `curr_queue_pairs - 1` queue pairs.
#[derive(Debug)]
struct TxWaitingRest;

impl VmStateInfo<VirtioNetVmState> for TxWaitingRest {
    fn name(&self) -> &'static str {
        "virtio-net-tx_waiting"
    }

    fn load(&self, f: &mut StreamReader<'_>, v: &mut VirtioNetVmState, _size: usize) -> Result<()> {
        if v.curr_queue_pairs > v.max_queue_pairs {
            bail!(
                "virtio-net: curr_queue_pairs {:x} > max_queue_pairs {:x}",
                v.curr_queue_pairs,
                v.max_queue_pairs
            );
        }
        let n = usize::from(v.curr_queue_pairs);
        if v.tx_waiting.len() < n {
            v.tx_waiting.resize(n, 0);
        }
        for w in v.tx_waiting.iter_mut().take(n).skip(1) {
            *w = f.get_be32();
        }
        Ok(())
    }

    fn save(&self, f: &mut StreamWriter, v: &VirtioNetVmState, _size: usize) -> Result<()> {
        for i in 1..usize::from(v.curr_queue_pairs) {
            f.put_be32(v.tx_waiting.get(i).copied().unwrap_or(0));
        }
        Ok(())
    }
}

/// `mac_table.in_use` and the guarded `mac_table.macs` pair: a table bigger than this one is
/// skipped and comes back as [`MAC_TABLE_ENTRIES`]` + 1` empty entries, which the device
/// model drops.
#[derive(Debug)]
struct MacTable;

impl VmStateInfo<Vec<[u8; 6]>> for MacTable {
    fn name(&self) -> &'static str {
        "mac_table"
    }

    fn load(&self, f: &mut StreamReader<'_>, v: &mut Vec<[u8; 6]>, _size: usize) -> Result<()> {
        let in_use = f.get_be32() as usize;
        if in_use > MAC_TABLE_ENTRIES {
            f.skip(in_use * 6);
            *v = vec![[0; 6]; MAC_TABLE_ENTRIES + 1];
            return Ok(());
        }
        v.clear();
        for _ in 0..in_use {
            let mut mac = [0; 6];
            f.get_buffer(&mut mac);
            v.push(mac);
        }
        Ok(())
    }

    fn save(&self, f: &mut StreamWriter, v: &Vec<[u8; 6]>, _size: usize) -> Result<()> {
        f.put_be32(v.len() as u32);
        for mac in v {
            f.put_buffer(mac);
        }
        Ok(())
    }
}

/// `vlans`, `VMSTATE_BUFFER_POINTER_UNSAFE`: the bitmap words in host (little endian) order.
#[derive(Debug)]
struct Vlans;

impl VmStateInfo<Vec<u32>> for Vlans {
    fn name(&self) -> &'static str {
        "buffer"
    }

    fn load(&self, f: &mut StreamReader<'_>, v: &mut Vec<u32>, _size: usize) -> Result<()> {
        let mut b = [0; VLAN_BYTES];
        f.get_buffer(&mut b);
        *v = b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        Ok(())
    }

    fn save(&self, f: &mut StreamWriter, v: &Vec<u32>, _size: usize) -> Result<()> {
        let mut b = [0; VLAN_BYTES];
        for (c, w) in b.chunks_exact_mut(4).zip(v) {
            c.copy_from_slice(&w.to_le_bytes());
        }
        f.put_buffer(&b);
        Ok(())
    }
}

/// `vmstate_virtio_net_device`. `virtio_net_post_load_device()` is in
/// [`VirtioNet::vmstate_load`]. The `rss` and vhost-user `backend` subsections are not
/// accepted: neither exists here.
static VMSTATE_VIRTIO_NET_DEVICE: LazyLock<VmStateDescription<NetDevice>> = LazyLock::new(|| {
    type N = NetDevice;
    VmStateDescription::new("virtio-net-device")
        .version_id(VIRTIO_NET_VM_VERSION)
        .minimum_version_id(VIRTIO_NET_VM_VERSION)
        .fields([
            VmStateField::array("mac", |n: &mut N| &mut n.s.mac),
            VmStateField::single("vqs", &TxWaitingFirst, |n: &mut N| &mut n.s.tx_waiting),
            VmStateField::scalar("mergeable_rx_bufs", |n: &mut N| &mut n.s.mergeable_rx_bufs),
            VmStateField::scalar("status", |n: &mut N| &mut n.s.status),
            VmStateField::scalar("promisc", |n: &mut N| &mut n.s.promisc),
            VmStateField::scalar("allmulti", |n: &mut N| &mut n.s.allmulti),
            VmStateField::single("mac_table", &MacTable, |n: &mut N| &mut n.s.macs),
            VmStateField::single("vlans", &Vlans, |n: &mut N| &mut n.s.vlans),
            VmStateField::scalar("has_vnet_hdr", |n: &mut N| &mut n.s.has_vnet_hdr),
            VmStateField::scalar("mac_table.multi_overflow", |n: &mut N| &mut n.s.multi_overflow),
            VmStateField::scalar("mac_table.uni_overflow", |n: &mut N| &mut n.s.uni_overflow),
            VmStateField::scalar("alluni", |n: &mut N| &mut n.s.alluni),
            VmStateField::scalar("nomulti", |n: &mut N| &mut n.s.nomulti),
            VmStateField::scalar("nouni", |n: &mut N| &mut n.s.nouni),
            VmStateField::scalar("nobcast", |n: &mut N| &mut n.s.nobcast),
            VmStateField::scalar("has_ufo", |n: &mut N| &mut n.s.has_ufo),
            VmStateField::single("max_queue_pairs", &Uint16Equal, |n: &mut N| {
                &mut n.s.max_queue_pairs
            })
            .test(|n: &N, _| n.s.max_queue_pairs > 1),
            VmStateField::scalar("curr_queue_pairs", |n: &mut N| &mut n.s.curr_queue_pairs)
                .test(|n: &N, _| n.s.max_queue_pairs > 1),
            VmStateField::single("tx_waiting", &TxWaitingRest, |n: &mut N| &mut n.s),
            VmStateField::scalar("curr_guest_offloads", |n: &mut N| &mut n.s.curr_guest_offloads)
                .test(|n: &N, _| n.ctrl_guest_offloads),
        ])
});

// virtio-balloon.

/// `vmstate_virtio_balloon_free_page_hint`.
static VMSTATE_BALLOON_FREE_PAGE_HINT: LazyLock<VmStateDescription<VirtioBalloonVmState>> =
    LazyLock::new(|| {
        type B = VirtioBalloonVmState;
        VmStateDescription::new("virtio-balloon-device/free-page-report")
            .version_id(1)
            .minimum_version_id(1)
            .needed(|b: &B| b.free_page_hint)
            .fields([
                VmStateField::scalar("free_page_hint_cmd_id", |b: &mut B| {
                    &mut b.free_page_hint_cmd_id
                }),
                VmStateField::scalar("free_page_hint_status", |b: &mut B| {
                    &mut b.free_page_hint_status
                }),
            ])
    });

/// `vmstate_virtio_balloon_page_poison`.
static VMSTATE_BALLOON_PAGE_POISON: LazyLock<VmStateDescription<VirtioBalloonVmState>> =
    LazyLock::new(|| {
        type B = VirtioBalloonVmState;
        VmStateDescription::new("virtio-balloon-device/page-poison")
            .version_id(1)
            .minimum_version_id(1)
            .needed(|b: &B| b.page_poison)
            .field(VmStateField::scalar("poison_val", |b: &mut B| &mut b.poison_val))
    });

/// `vmstate_virtio_balloon_device`. `virtio_balloon_post_load_device()` is in the class
/// `post_load`.
static VMSTATE_BALLOON_DEVICE: LazyLock<VmStateDescription<VirtioBalloonVmState>> =
    LazyLock::new(|| {
        type B = VirtioBalloonVmState;
        VmStateDescription::new("virtio-balloon-device")
            .version_id(1)
            .minimum_version_id(1)
            .fields([
                VmStateField::scalar("num_pages", |b: &mut B| &mut b.num_pages),
                VmStateField::scalar("actual", |b: &mut B| &mut b.actual),
            ])
            .subsection(&VMSTATE_BALLOON_FREE_PAGE_HINT)
            .subsection(&VMSTATE_BALLOON_PAGE_POISON)
    });

// virtio-serial.

/// `virtio_serial_save_device()`.
fn console_save(f: &mut StreamWriter, c: &VirtioConsoleVmState) -> Result<()> {
    f.put_be16(c.cols);
    f.put_be16(c.rows);
    f.put_be32(c.max_nr_ports);
    for w in &c.ports_map {
        f.put_be32(*w);
    }
    f.put_be32(c.ports.len() as u32);
    for p in &c.ports {
        if p.elem_popped != 0 {
            bail!("virtio-serial: a port holds a guest buffer");
        }
        f.put_be32(p.id);
        f.put_byte(p.guest_connected);
        f.put_byte(p.host_connected);
        f.put_be32(0);
    }
    Ok(())
}

/// `virtio_serial_load_device()`. The ports map has as many words as this end's, which come
/// with `c`; the device model checks the values.
fn console_load(f: &mut StreamReader<'_>, c: &mut VirtioConsoleVmState) -> Result<()> {
    c.cols = f.get_be16();
    c.rows = f.get_be16();
    c.max_nr_ports = f.get_be32();
    for w in &mut c.ports_map {
        *w = f.get_be32();
    }
    let nr_active = f.get_be32();
    // One bit of the map per active port.
    if nr_active as usize > c.ports_map.len() * 32 {
        bail!("virtio-serial: {nr_active} active ports");
    }
    c.ports.clear();
    for _ in 0..nr_active {
        let id = f.get_be32();
        let guest_connected = f.get_byte();
        let host_connected = f.get_byte();
        let elem_popped = f.get_be32();
        if elem_popped != 0 {
            bail!("virtio-serial: port {id} holds a guest buffer, which is not supported");
        }
        c.ports.push(VirtioConsolePortVmState { id, guest_connected, host_connected, elem_popped });
    }
    Ok(())
}

// The device section.

/// `virtio_save()`.
fn virtio_save(f: &mut StreamWriter, s: &mut VirtioSection) -> Result<()> {
    let (mmio, msix) = (s.mmio(), s.msix());
    match &mut s.transport {
        // virtio_pci_save_config(): pci_device_save(), msix_save() and the config vector.
        Transport::Pci(p) => {
            vmstate_save_state(f, &VMSTATE_PCI_DEVICE, &mut p.pci)?;
            if msix {
                f.put_buffer(&p.msix_table);
                f.put_buffer(&p.msix_pba);
                f.put_be16(p.vdev.config_vector);
            }
        }
        Transport::Mmio(m) => {
            f.put_be32(m.host_features_sel);
            f.put_be32(m.guest_features_sel);
            f.put_be32(m.guest_page_shift);
        }
    }

    let vdev = s.vdev_ref();
    f.put_byte(vdev.status);
    f.put_byte(vdev.isr);
    f.put_be16(vdev.queue_sel);
    f.put_be32(vdev.guest_features as u32);
    f.put_be32(vdev.config.len() as u32);
    f.put_buffer(&vdev.config);
    let nvqs = vdev.nvqs.min(vdev.vqs.len());
    f.put_be32(nvqs as u32);
    for q in &vdev.vqs[..nvqs] {
        f.put_be32(q.num);
        if mmio {
            f.put_be32(q.align);
        }
        f.put_be64(q.desc);
        f.put_be16(q.last_avail_idx);
        if msix {
            f.put_be16(q.vector);
        }
    }

    match &mut s.model {
        VirtioModel::Blk => f.put_byte(0),
        VirtioModel::Rng | VirtioModel::Scsi => {}
        VirtioModel::Net(n) => vmstate_save_state(f, &VMSTATE_VIRTIO_NET_DEVICE, n)?,
        VirtioModel::Balloon(b) => vmstate_save_state(f, &VMSTATE_BALLOON_DEVICE, b)?,
        VirtioModel::Console(c) => console_save(f, c)?,
    }
    vmstate_save_state(f, &VMSTATE_VIRTIO, s)
}

/// `virtio_load()` up to the point where the state goes back into the device: the stream into
/// `s`, which starts out as this end's state so it has this end's sizes.
fn virtio_load(f: &mut StreamReader<'_>, s: &mut VirtioSection) -> Result<()> {
    let (mmio, msix) = (s.mmio(), s.msix());
    match &mut s.transport {
        // virtio_pci_load_config(): pci_device_load(), msix_load() and the config vector.
        Transport::Pci(p) => {
            vmstate_load_state(f, &VMSTATE_PCI_DEVICE, &mut p.pci, 2)?;
            if msix {
                f.get_buffer(&mut p.msix_table);
                f.get_buffer(&mut p.msix_pba);
                p.vdev.config_vector = f.get_be16();
            } else {
                p.vdev.config_vector = VIRTIO_NO_VECTOR;
            }
        }
        Transport::Mmio(m) => {
            m.host_features_sel = f.get_be32();
            m.guest_features_sel = f.get_be32();
            m.guest_page_shift = f.get_be32();
        }
    }

    let vdev = s.vdev();
    vdev.device_endian = VIRTIO_DEVICE_ENDIAN_LITTLE;
    vdev.status = f.get_byte();
    vdev.isr = f.get_byte();
    vdev.queue_sel = f.get_be16();
    if usize::from(vdev.queue_sel) >= VIRTIO_QUEUE_MAX {
        bail!("virtio: queue_sel {} out of range", vdev.queue_sel);
    }
    // The low half for now, as QEMU has it while the device model loads.
    vdev.guest_features = u64::from(f.get_be32());
    vdev.guest_features_hi = 0;
    let config_len = f.get_be32() as usize;
    let n = config_len.min(vdev.config.len());
    f.get_buffer(&mut vdev.config[..n]);
    f.skip(config_len - n);
    if f.get_error() != 0 {
        bail!("virtio: stream ended in the config space");
    }
    let num = f.get_be32() as usize;
    if num > VIRTIO_QUEUE_MAX {
        bail!("Invalid number of virtqueues: {num:#x}");
    }
    vdev.resize_queues(num);
    for q in &mut vdev.vqs[..num] {
        q.num = f.get_be32();
        if mmio {
            q.align = f.get_be32();
        }
        q.desc = f.get_be64();
        q.last_avail_idx = f.get_be16();
        q.vector = if msix { f.get_be16() } else { VIRTIO_NO_VECTOR };
    }
    let features = vdev.guest_features;

    match &mut s.model {
        VirtioModel::Blk => {
            if f.get_byte() != 0 {
                bail!("virtio-blk: requests in flight are not supported");
            }
        }
        VirtioModel::Rng | VirtioModel::Scsi => {}
        VirtioModel::Net(n) => {
            n.ctrl_guest_offloads = has_feature(features, VIRTIO_NET_F_CTRL_GUEST_OFFLOADS);
            vmstate_load_state(f, &VMSTATE_VIRTIO_NET_DEVICE, n, VIRTIO_NET_VM_VERSION)?;
        }
        VirtioModel::Balloon(b) => vmstate_load_state(f, &VMSTATE_BALLOON_DEVICE, b, 1)?,
        VirtioModel::Console(c) => console_load(f, c)?,
    }
    vmstate_load_state(f, &VMSTATE_VIRTIO, s, 1)
}

/// `VMSTATE_VIRTIO_DEVICE`: `virtio_device_put()` and `virtio_device_get()`.
#[derive(Debug)]
struct VirtioDevice;

impl VmStateInfo<VirtioSection> for VirtioDevice {
    fn name(&self) -> &'static str {
        "virtio"
    }

    fn load(&self, f: &mut StreamReader<'_>, v: &mut VirtioSection, _size: usize) -> Result<()> {
        v.pad();
        virtio_load(f, v)
    }

    fn save(&self, f: &mut StreamWriter, v: &VirtioSection, _size: usize) -> Result<()> {
        let mut s = v.clone();
        s.pad();
        virtio_save(f, &mut s)
    }
}

fn device_vmsd(name: &'static str, version: i32) -> VmStateDescription<VirtioSection> {
    VmStateDescription::new(name)
        .version_id(version)
        .minimum_version_id(version)
        .field(VmStateField::single("virtio", &VirtioDevice, |s: &mut S| s))
}

static VMSTATE_VIRTIO_BLK: LazyLock<VmStateDescription<S>> =
    LazyLock::new(|| device_vmsd("virtio-blk", 2));
static VMSTATE_VIRTIO_RNG: LazyLock<VmStateDescription<S>> =
    LazyLock::new(|| device_vmsd("virtio-rng", 1));
static VMSTATE_VIRTIO_SCSI: LazyLock<VmStateDescription<S>> =
    LazyLock::new(|| device_vmsd("virtio-scsi", 1));
static VMSTATE_VIRTIO_NET: LazyLock<VmStateDescription<S>> =
    LazyLock::new(|| device_vmsd("virtio-net", VIRTIO_NET_VM_VERSION));
static VMSTATE_VIRTIO_BALLOON: LazyLock<VmStateDescription<S>> =
    LazyLock::new(|| device_vmsd("virtio-balloon", 1));
// 'console' is used for backwards compatibility.
static VMSTATE_VIRTIO_CONSOLE: LazyLock<VmStateDescription<S>> =
    LazyLock::new(|| device_vmsd("virtio-console", 3));

// scsi-disk.

/// `vmstate_info_scsi_requests`: the requests of a device, none here.
#[derive(Debug)]
struct ScsiRequests;

impl VmStateInfo<ScsiDiskVmState> for ScsiRequests {
    fn name(&self) -> &'static str {
        "scsi-requests"
    }

    fn load(&self, f: &mut StreamReader<'_>, _v: &mut ScsiDiskVmState, _size: usize) -> Result<()> {
        if f.get_byte() != 0 {
            bail!("scsi-disk: requests in flight are not supported");
        }
        Ok(())
    }

    fn save(&self, f: &mut StreamWriter, _v: &ScsiDiskVmState, _size: usize) -> Result<()> {
        f.put_byte(0);
        Ok(())
    }
}

/// `vmstate_scsi_sense_state`.
static VMSTATE_SCSI_SENSE_STATE: LazyLock<VmStateDescription<ScsiDiskVmState>> =
    LazyLock::new(|| {
        type D = ScsiDiskVmState;
        VmStateDescription::new("SCSIDevice/sense")
            .version_id(1)
            .minimum_version_id(1)
            .needed(|d: &D| d.sense_len as usize > SCSI_SENSE_BUF_SIZE_OLD)
            .field(VmStateField::varray(
                "sense",
                |_: &D| SCSI_SENSE_BUF_SIZE - SCSI_SENSE_BUF_SIZE_OLD,
                |d: &mut D| &mut d.sense[SCSI_SENSE_BUF_SIZE_OLD..],
            ))
    });

/// `vmstate_scsi_device`.
static VMSTATE_SCSI_DEVICE: LazyLock<VmStateDescription<ScsiDiskVmState>> = LazyLock::new(|| {
    type D = ScsiDiskVmState;
    VmStateDescription::new("SCSIDevice")
        .version_id(1)
        .minimum_version_id(1)
        .fields([
            VmStateField::scalar("unit_attention.key", |d: &mut D| &mut d.unit_attention[0]),
            VmStateField::scalar("unit_attention.asc", |d: &mut D| &mut d.unit_attention[1]),
            VmStateField::scalar("unit_attention.ascq", |d: &mut D| &mut d.unit_attention[2]),
            VmStateField::scalar("sense_is_ua", |d: &mut D| &mut d.sense_is_ua),
            VmStateField::varray(
                "sense",
                |_: &D| SCSI_SENSE_BUF_SIZE_OLD,
                |d: &mut D| &mut d.sense[..SCSI_SENSE_BUF_SIZE_OLD],
            ),
            VmStateField::scalar("sense_len", |d: &mut D| &mut d.sense_len),
            VmStateField::single("requests", &ScsiRequests, |d: &mut D| d),
        ])
        .subsection(&VMSTATE_SCSI_SENSE_STATE)
});

/// `vmstate_scsi_disk_state`, for scsi-hd and scsi-cd alike.
static VMSTATE_SCSI_DISK_STATE: LazyLock<VmStateDescription<ScsiDiskVmState>> =
    LazyLock::new(|| {
        type D = ScsiDiskVmState;
        VmStateDescription::new("scsi-disk").version_id(1).minimum_version_id(1).fields([
            VmStateField::structure("qdev", &VMSTATE_SCSI_DEVICE, |d: &mut D| d),
            VmStateField::scalar("media_changed", |d: &mut D| &mut d.media_changed),
            VmStateField::scalar("media_event", |d: &mut D| &mut d.media_event),
            VmStateField::scalar("eject_request", |d: &mut D| &mut d.eject_request),
            VmStateField::scalar("tray_open", |d: &mut D| &mut d.tray_open),
            VmStateField::scalar("tray_locked", |d: &mut D| &mut d.tray_locked),
        ])
    });

// Registration.

/// The section description for a device ID, `None` for a device without migration support.
fn vmsd_for(device_id: u16) -> Option<&'static VmStateDescription<S>> {
    Some(match device_id {
        VIRTIO_ID_BLOCK => &VMSTATE_VIRTIO_BLK,
        VIRTIO_ID_RNG => &VMSTATE_VIRTIO_RNG,
        VIRTIO_ID_SCSI => &VMSTATE_VIRTIO_SCSI,
        VIRTIO_ID_NET => &VMSTATE_VIRTIO_NET,
        VIRTIO_ID_BALLOON => &VMSTATE_VIRTIO_BALLOON,
        VIRTIO_ID_CONSOLE => &VMSTATE_VIRTIO_CONSOLE,
        _ => return None,
    })
}

/// The device model's state.
fn model_save(b: &mut VirtioBackend) -> Result<VirtioModel> {
    let id = b.vdev().device_id();
    Ok(match id {
        VIRTIO_ID_BLOCK => VirtioModel::Blk,
        VIRTIO_ID_RNG => VirtioModel::Rng,
        VIRTIO_ID_SCSI => VirtioModel::Scsi,
        VIRTIO_ID_NET => {
            let (vdev, n) =
                b.downcast_mut::<VirtioNet>().ok_or_else(|| err!("virtio-net: wrong model"))?;
            VirtioModel::Net(NetDevice {
                s: n.vmstate_save(),
                ctrl_guest_offloads: vdev.has_feature(VIRTIO_NET_F_CTRL_GUEST_OFFLOADS),
            })
        }
        VIRTIO_ID_BALLOON => {
            let (vdev, d) = b
                .downcast_mut::<VirtioBalloon>()
                .ok_or_else(|| err!("virtio-balloon: wrong model"))?;
            VirtioModel::Balloon(d.vmstate_save(vdev))
        }
        VIRTIO_ID_CONSOLE => {
            let (_, d) = b
                .downcast_mut::<VirtioConsole>()
                .ok_or_else(|| err!("virtio-serial: wrong model"))?;
            VirtioModel::Console(d.vmstate_save())
        }
        _ => bail!("virtio device {id} cannot be migrated"),
    })
}

/// Puts the device model's state back, the `device` step of the transport's load.
fn model_load(
    model: &VirtioModel,
    vdev: &mut VirtIODevice,
    class: &mut dyn VirtioDeviceClass,
) -> Result<()> {
    let any = class.as_any_mut();
    match model {
        VirtioModel::Blk | VirtioModel::Rng | VirtioModel::Scsi => Ok(()),
        VirtioModel::Net(n) => any
            .downcast_mut::<VirtioNet>()
            .ok_or_else(|| err!("virtio-net: wrong model"))?
            .vmstate_load(vdev, &n.s),
        VirtioModel::Balloon(s) => any
            .downcast_mut::<VirtioBalloon>()
            .ok_or_else(|| err!("virtio-balloon: wrong model"))?
            .vmstate_load(s),
        VirtioModel::Console(s) => any
            .downcast_mut::<VirtioConsole>()
            .ok_or_else(|| err!("virtio-serial: wrong model"))?
            .vmstate_load(s),
    }
}

/// What a transport gives the registration.
trait Proxy: Send + Sync + 'static {
    fn backend<R>(&self, f: impl FnOnce(&mut VirtioBackend) -> R) -> Option<R>;
    fn save(&self) -> Option<Transport>;
    fn load(
        &self,
        t: &Transport,
        device: impl FnOnce(&mut VirtIODevice, &mut dyn VirtioDeviceClass) -> Result<()>,
    ) -> Result<()>;
}

impl Proxy for VirtioPci {
    fn backend<R>(&self, f: impl FnOnce(&mut VirtioBackend) -> R) -> Option<R> {
        self.with_backend(f)
    }

    fn save(&self) -> Option<Transport> {
        self.vmstate_save().map(|s| Transport::Pci(Box::new(s)))
    }

    fn load(
        &self,
        t: &Transport,
        device: impl FnOnce(&mut VirtIODevice, &mut dyn VirtioDeviceClass) -> Result<()>,
    ) -> Result<()> {
        match t {
            Transport::Pci(s) => self.vmstate_load(s, device),
            Transport::Mmio(_) => bail!("virtio-pci: virtio-mmio state"),
        }
    }
}

impl Proxy for VirtioMmio {
    fn backend<R>(&self, f: impl FnOnce(&mut VirtioBackend) -> R) -> Option<R> {
        self.with_backend(f)
    }

    fn save(&self) -> Option<Transport> {
        self.vmstate_save().map(|s| Transport::Mmio(Box::new(s)))
    }

    fn load(
        &self,
        t: &Transport,
        device: impl FnOnce(&mut VirtIODevice, &mut dyn VirtioDeviceClass) -> Result<()>,
    ) -> Result<()> {
        match t {
            Transport::Mmio(s) => self.vmstate_load(s, device),
            Transport::Pci(_) => bail!("virtio-mmio: virtio-pci state"),
        }
    }
}

fn section_save<P: Proxy>(p: &P) -> Result<VirtioSection> {
    let transport = p.save().ok_or_else(|| err!("virtio: device busy or missing"))?;
    let model = p.backend(model_save).ok_or_else(|| err!("virtio: device busy or missing"))??;
    Ok(VirtioSection { transport, model })
}

fn section_load<P: Proxy>(p: &P, s: &VirtioSection) -> Result<()> {
    p.load(&s.transport, |vdev, class| model_load(&s.model, vdev, class))
}

/// Registers the section of the device behind `proxy`, whose path is `path`, and the
/// sections of its `scsi-disk` devices if it is a virtio-scsi.
fn register<P: Proxy>(savevm: &mut SaveVm, path: &str, proxy: Arc<P>) {
    let Some(id) = proxy.backend(|b| b.vdev().device_id()) else {
        return;
    };
    let Some(vmsd) = vmsd_for(id) else {
        return;
    };
    let (get, put) = (Arc::clone(&proxy), Arc::clone(&proxy));
    savevm.register_vmsd(
        &format!("{path}/"),
        Some(0),
        vmsd,
        move || section_save(&*get),
        move |s| section_load(&*put, &s),
    );
    if id != VIRTIO_ID_SCSI {
        return;
    }
    let disks = proxy
        .backend(|b| {
            b.downcast_mut::<VirtioScsi>()
                .map(|(_, s)| s.bus().devices().iter().map(|d| d.address()).collect::<Vec<_>>())
        })
        .flatten()
        .unwrap_or_default();
    for (i, (channel, scsi_id, lun)) in disks.into_iter().enumerate() {
        let (get, put) = (Arc::clone(&proxy), Arc::clone(&proxy));
        savevm.register_vmsd(
            &format!("{path}/{channel}:{scsi_id}:{lun}/"),
            Some(0),
            &VMSTATE_SCSI_DISK_STATE,
            move || {
                get.backend(|b| {
                    b.downcast_mut::<VirtioScsi>()
                        .and_then(|(_, s)| s.bus().device(i).map(|d| d.vmstate_save()))
                })
                .flatten()
                .ok_or_else(|| err!("scsi-disk {i}: device busy or missing"))
            },
            move |s| {
                put.backend(|b| {
                    b.downcast_mut::<VirtioScsi>()
                        .and_then(|(_, scsi)| scsi.bus_mut().device_mut(i))
                        .map(|d| d.vmstate_load(&s))
                })
                .flatten()
                .ok_or_else(|| err!("scsi-disk {i}: device busy or missing"))?
            },
        );
    }
}

/// Registers the virtio PCI function `dev` on the root bus, `0000:00:SS.F/<vmsd>`.
pub(crate) fn register_pci(savevm: &mut SaveVm, dev: &VirtioPci) {
    let devfn = dev.pci_dev().devfn();
    let path = format!("0000:00:{:02x}.{:x}", devfn >> 3, devfn & 7);
    register(savevm, &path, Arc::new(dev.clone()));
}

/// Registers the device on the virtio-mmio transport at `base`, `virtio-mmio@<base>/<vmsd>`.
pub(crate) fn register_mmio(savevm: &mut SaveVm, base: u64, t: Arc<VirtioMmio>) {
    register(savevm, &format!("virtio-mmio@{base:016x}"), t);
}

#[cfg(test)]
mod tests {
    use ruvm_hw_pci::PciDeviceVmState;

    use super::*;

    fn vdev(nq: usize, config_len: usize) -> VirtioVmState {
        let mut vqs = vec![VirtQueueVmState::default(); nq];
        for (i, q) in vqs.iter_mut().enumerate() {
            q.num = 256;
            q.num_default = 256;
            q.desc = 0x1000 * (i as u64 + 1);
            q.avail = q.desc + 0x1000;
            q.used = q.desc + 0x2000;
            q.last_avail_idx = 7 + i as u16;
            q.vector = i as u16 + 1;
        }
        VirtioVmState {
            device_id: VIRTIO_ID_BLOCK,
            host_features: 1 << VIRTIO_F_VERSION_1 | 0x7000_0000,
            status: 0x0f,
            isr: 0,
            queue_sel: 0,
            guest_features: 1 << VIRTIO_F_VERSION_1 | 0x1000_0000,
            guest_features_hi: 0,
            config: (0..config_len as u8).collect(),
            nvqs: nq,
            vqs,
            device_endian: VIRTIO_DEVICE_ENDIAN_LITTLE,
            broken: false,
            started: true,
            disabled: false,
            config_vector: 0,
        }
    }

    fn pci(model: VirtioModel, nq: usize, config_len: usize) -> VirtioSection {
        let mut config = vec![0; 256];
        config[0] = 0xf4;
        config[1] = 0x1a;
        let mut vqs = vec![VirtioPciQueueVmState::default(); VIRTIO_QUEUE_MAX];
        vqs[0] = VirtioPciQueueVmState {
            num: 256,
            enabled: true,
            desc: [0x1000, 0],
            avail: [0x2000, 0],
            used: [0x3000, 0],
        };
        VirtioSection {
            transport: Transport::Pci(Box::new(VirtioPciVmState {
                pci: PciDeviceVmState { version_id: 2, config, irq_state: [0; 4] },
                msix_vectors: 2,
                msix_table: vec![0x11; 32],
                msix_pba: vec![0],
                modern: true,
                dfselect: 1,
                gfselect: 1,
                guest_features: [0x1000_0000, 1, 0, 0],
                vqs,
                vdev: vdev(nq, config_len),
            })),
            model,
        }
    }

    fn save(s: &mut VirtioSection, vmsd: &VmStateDescription<S>) -> Vec<u8> {
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, vmsd, s).unwrap();
        f.into_inner()
    }

    /// Loads `b` into a copy of `like` with the migrated values cleared.
    fn load(b: &[u8], like: &VirtioSection, vmsd: &VmStateDescription<S>) -> Result<S> {
        let mut back = like.clone();
        let v = back.vdev();
        v.status = 0;
        v.guest_features = 0;
        v.config.iter_mut().for_each(|c| *c = 0);
        for q in &mut v.vqs {
            q.desc = 0;
            q.avail = 0;
            q.last_avail_idx = 0;
        }
        // A section footer follows in a real stream.
        let b = [b, &[0x7e]].concat();
        vmstate_load_state(&mut StreamReader::new(&b), vmsd, &mut back, vmsd.version_id)?;
        Ok(back)
    }

    /// The length of a subsection header.
    fn sub(name: &str) -> usize {
        1 + 1 + name.len() + 4
    }

    #[test]
    fn virtio_blk_pci_layout_matches_qemu() {
        let mut s = pci(VirtioModel::Blk, 1, 60);
        let b = save(&mut s, &VMSTATE_VIRTIO_BLK);
        let pci_config = 4 + 256 + 16 + 32 + 1 + 2;
        let core = 1 + 1 + 2 + 4 + 4 + 60 + 4 + (4 + 8 + 2 + 2);
        let blk = 1;
        let subs = sub("virtio/64bit_features")
            + 8
            + sub("virtio/virtqueues")
            + VIRTIO_QUEUE_MAX * 16
            + sub("virtio/extra_state")
            + sub("virtio_pci/modern_state")
            + 4
            + 4
            + 8
            + VIRTIO_QUEUE_MAX * 28
            + sub("virtio/started")
            + 1;
        assert_eq!(b.len(), pci_config + core + blk + subs);
        // The low feature word after status, isr and queue_sel.
        assert_eq!(&b[pci_config + 4..pci_config + 8], &0x1000_0000u32.to_be_bytes());
        // The queue: num, desc, last_avail_idx and the MSI-X vector.
        let q = pci_config + 1 + 1 + 2 + 4 + 4 + 60 + 4;
        assert_eq!(&b[q..q + 4], &256u32.to_be_bytes());
        assert_eq!(&b[q + 12..q + 16], &[0, 7, 0, 1]);
        assert_eq!(b[q + 16], 0, "no blk requests");

        let mut back = load(&b, &s, &VMSTATE_VIRTIO_BLK).unwrap();
        back.pad();
        s.pad();
        assert_eq!(back.vdev_ref(), s.vdev_ref());
        match (&back.transport, &s.transport) {
            (Transport::Pci(a), Transport::Pci(b)) => assert_eq!(a, b),
            _ => unreachable!(),
        }
    }

    #[test]
    fn virtio_mmio_legacy_has_no_extra_state() {
        let mut v = vdev(1, 8);
        v.host_features = 0x7000_0000;
        v.guest_features = 0x1000_0000;
        let mut s = VirtioSection {
            transport: Transport::Mmio(Box::new(VirtioMmioVmState {
                guest_page_shift: 12,
                legacy: true,
                vqs: vec![VirtioMmioQueueVmState::default(); VIRTIO_QUEUE_MAX],
                vdev: v,
                ..VirtioMmioVmState::default()
            })),
            model: VirtioModel::Rng,
        };
        s.vdev().vqs[0].align = 4096;
        let b = save(&mut s, &VMSTATE_VIRTIO_RNG);
        // The transport words, the core, one queue with its alignment, then started.
        assert_eq!(b.len(), 12 + 12 + 8 + 4 + 4 + 4 + 8 + 2 + sub("virtio/started") + 1);
        let back = load(&b, &s, &VMSTATE_VIRTIO_RNG).unwrap();
        assert_eq!(back.vdev_ref().vqs[0].align, 4096);
        assert_eq!(back.vdev_ref().vqs[0].desc, 0x1000);

        // A modern transport sends its queue registers.
        let Transport::Mmio(m) = &mut s.transport else { unreachable!() };
        m.legacy = false;
        let b2 = save(&mut s, &VMSTATE_VIRTIO_RNG);
        let extra =
            sub("virtio/extra_state") + sub("virtio_mmio/state") + 8 + VIRTIO_QUEUE_MAX * 27;
        assert_eq!(b2.len(), b.len() + extra);
        load(&b2, &s, &VMSTATE_VIRTIO_RNG).unwrap();
    }

    #[test]
    fn virtio_net_device_fields() {
        let net = VirtioNetVmState {
            mac: [0x52, 0x54, 0, 0x12, 0x34, 0x56],
            tx_waiting: vec![0],
            mergeable_rx_bufs: 1,
            status: 1,
            macs: vec![[1, 2, 3, 4, 5, 6]],
            vlans: vec![0; 128],
            max_queue_pairs: 1,
            curr_queue_pairs: 1,
            curr_guest_offloads: 0x1f,
            ..VirtioNetVmState::default()
        };
        let model = VirtioModel::Net(NetDevice { s: net, ctrl_guest_offloads: false });
        let mut s = pci(model, 3, 12);
        s.vdev().guest_features |= 1 << VIRTIO_NET_F_CTRL_GUEST_OFFLOADS;
        let mut plain = s.clone();
        plain.vdev().guest_features &= !(1 << VIRTIO_NET_F_CTRL_GUEST_OFFLOADS);
        let with = save(&mut s, &VMSTATE_VIRTIO_NET);
        let without = save(&mut plain, &VMSTATE_VIRTIO_NET);
        // curr_guest_offloads goes with the feature, as the saved model state says.
        let VirtioModel::Net(n) = &mut s.model else { unreachable!() };
        n.ctrl_guest_offloads = true;
        let with =
            if with.len() == without.len() { save(&mut s, &VMSTATE_VIRTIO_NET) } else { with };
        assert_eq!(with.len(), without.len() + 8);
        let net_len = 6 + 4 + 4 + 2 + 1 + 1 + 4 + 6 + VLAN_BYTES + 4 + 6 + 1;
        assert_eq!(
            without.len() - save(&mut pci(VirtioModel::Rng, 3, 12), &VMSTATE_VIRTIO_RNG).len(),
            net_len
        );

        let back = load(&with, &s, &VMSTATE_VIRTIO_NET).unwrap();
        let (VirtioModel::Net(a), VirtioModel::Net(b)) = (&back.model, &s.model) else {
            unreachable!()
        };
        assert_eq!(a.s, b.s);
        assert!(a.ctrl_guest_offloads);
    }

    #[test]
    fn virtio_net_multiqueue_and_big_mac_table() {
        let net = VirtioNetVmState {
            tx_waiting: vec![1, 0, 1, 0],
            macs: vec![[0; 6]; 70],
            vlans: vec![0; 128],
            max_queue_pairs: 4,
            curr_queue_pairs: 3,
            ..VirtioNetVmState::default()
        };
        let mut s = pci(VirtioModel::Net(NetDevice { s: net, ctrl_guest_offloads: false }), 9, 12);
        let b = save(&mut s, &VMSTATE_VIRTIO_NET);
        let back = load(&b, &s, &VMSTATE_VIRTIO_NET).unwrap();
        let VirtioModel::Net(n) = &back.model else { unreachable!() };
        assert_eq!(n.s.tx_waiting, vec![1, 0, 1, 0]);
        assert_eq!(n.s.curr_queue_pairs, 3);
        assert_eq!(n.s.macs.len(), MAC_TABLE_ENTRIES + 1);

        // max_queue_pairs must match.
        let mut other = s.clone();
        let VirtioModel::Net(n) = &mut other.model else { unreachable!() };
        n.s.max_queue_pairs = 2;
        n.s.tx_waiting.truncate(2);
        assert!(load(&b, &other, &VMSTATE_VIRTIO_NET).is_err());
    }

    #[test]
    fn virtio_balloon_and_console() {
        let bal = VirtioBalloonVmState {
            num_pages: 0x100,
            actual: 0x80,
            free_page_hint: true,
            free_page_hint_cmd_id: 5,
            free_page_hint_status: 2,
            ..VirtioBalloonVmState::default()
        };
        let mut s = pci(VirtioModel::Balloon(bal), 3, 8);
        let base = save(&mut pci(VirtioModel::Rng, 3, 8), &VMSTATE_VIRTIO_RNG).len();
        let b = save(&mut s, &VMSTATE_VIRTIO_BALLOON);
        assert_eq!(b.len(), base + 8 + sub("virtio-balloon-device/free-page-report") + 8);
        let back = load(&b, &s, &VMSTATE_VIRTIO_BALLOON).unwrap();
        let (VirtioModel::Balloon(x), VirtioModel::Balloon(y)) = (&back.model, &s.model) else {
            unreachable!()
        };
        assert_eq!(x, y);

        let con = VirtioConsoleVmState {
            ports: vec![VirtioConsolePortVmState {
                id: 0,
                guest_connected: 1,
                host_connected: 1,
                elem_popped: 0,
            }],
            ..VirtioConsoleVmState::default()
        };
        let mut s = pci(VirtioModel::Console(con), 2, 12);
        let base = save(&mut pci(VirtioModel::Rng, 2, 12), &VMSTATE_VIRTIO_RNG).len();
        let b = save(&mut s, &VMSTATE_VIRTIO_CONSOLE);
        assert_eq!(b.len(), base + 2 + 2 + 4 + 4 + 4 + 10);
        let back = load(&b, &s, &VMSTATE_VIRTIO_CONSOLE).unwrap();
        let (VirtioModel::Console(x), VirtioModel::Console(y)) = (&back.model, &s.model) else {
            unreachable!()
        };
        assert_eq!(x, y);
    }

    #[test]
    fn scsi_disk_layout_matches_qemu() {
        let mut d = ScsiDiskVmState {
            sense: vec![0; SCSI_SENSE_BUF_SIZE],
            tray_locked: true,
            ..ScsiDiskVmState::default()
        };
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_SCSI_DISK_STATE, &mut d).unwrap();
        let b = f.into_inner();
        assert_eq!(b.len(), 3 + 1 + 96 + 4 + 1 + 5);
        assert_eq!(b[b.len() - 1], 1);

        d.sense_len = 200;
        d.sense[150] = 0x42;
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_SCSI_DISK_STATE, &mut d).unwrap();
        let b = f.into_inner();
        assert_eq!(b.len(), 3 + 1 + 96 + 4 + 1 + sub("SCSIDevice/sense") + 156 + 5);
        let mut back =
            ScsiDiskVmState { sense: vec![0; SCSI_SENSE_BUF_SIZE], ..Default::default() };
        let b = [&b[..], &[0x7e]].concat();
        vmstate_load_state(&mut StreamReader::new(&b), &VMSTATE_SCSI_DISK_STATE, &mut back, 1)
            .unwrap();
        assert_eq!(back, d);
    }

    #[test]
    fn bad_streams_fail() {
        let mut s = pci(VirtioModel::Blk, 1, 60);
        let mut b = save(&mut s, &VMSTATE_VIRTIO_BLK);
        // A pending blk request.
        let req = 4 + 256 + 16 + 32 + 1 + 2 + 1 + 1 + 2 + 4 + 4 + 60 + 4 + 16;
        b[req] = 1;
        assert!(load(&b, &s, &VMSTATE_VIRTIO_BLK).is_err());
        // Too many queues.
        let mut b = save(&mut s, &VMSTATE_VIRTIO_BLK);
        let nq = req - 4 - 16;
        b[nq..nq + 4].copy_from_slice(&0x401u32.to_be_bytes());
        assert!(load(&b, &s, &VMSTATE_VIRTIO_BLK).is_err());
    }
}
