// SPDX-License-Identifier: GPL-2.0-or-later

//! The virtio-pci transport, a port of `hw/virtio/virtio-pci.c`.
//!
//! [`VirtioPci::new`] registers a PCI function on a [`PciBus`] and plugs a realized
//! [`VirtioBackend`] into it. The BAR layout is QEMU's:
//!
//! - BAR0: the legacy I/O window, only for transitional devices (legacy enabled).
//! - BAR1: the MSI-X table and PBA, when `vectors` is not zero and the platform has MSI.
//! - BAR2: the optional `modern-pio-notify` I/O window.
//! - BAR4: a 64-bit prefetchable memory BAR holding the virtio 1.0 structures: common config
//!   at 0x0, ISR at 0x1000, device config at 0x2000 and the notify area at 0x3000.
//!
//! The virtio 1.0 structures are described by vendor capabilities in config space, followed
//! by the `VIRTIO_PCI_CAP_PCI_CFG` window that lets a driver reach BAR4 through config cycles.
//!
//! `disable-legacy` defaults to auto, which turns legacy off only when the device sits behind a
//! PCI Express port (a non-root bus whose bridge is an express function). A device on the root
//! bus of q35 or i440fx is therefore transitional by default, as in QEMU.
//!
//! Differences from QEMU:
//!
//! - There is no PCI Express endpoint capability, AER, ATS or FLR, since `ruvm-hw-pci` does not
//!   have them. A device behind an express port still gets the power management capability and
//!   4 KiB of config space.
//! - The config access window (`VIRTIO_PCI_CAP_PCI_CFG`) calls the region handlers directly
//!   instead of going through a private address space.
//! - Legacy and modern windows are always little endian.
//! - A register access from inside the device's own queue processing is dropped, like the
//!   memory re-entrancy guard in QEMU does.
//! - `virtio_pci_optimal_num_queues()` takes the vCPU count as an argument.
//!
//! Migration state is in the `vmstate` submodule, [`VirtioPciVmState`].
//!
//! Not ported: QOM registration and the property parser, ioeventfd and irqfd (every
//! notify is handled synchronously), KVM MSI routes, vhost vector masking, shared memory
//! capabilities, `VIRTIO_F_NOTIFICATION_DATA`, the extended (above 64 bit) feature words,
//! `bad_features` class hooks (a legacy driver that sets `VIRTIO_F_BAD_FEATURE` gets no
//! features), and the ACPI and bus-reset hooks of the proxy.

use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError, Weak};
use std::thread::{self, ThreadId};

use ruvm_base::{Error, Result, warn_report};
use ruvm_hw_pci::regs::{
    PCI_BASE_ADDRESS_MEM_PREFETCH, PCI_BASE_ADDRESS_MEM_TYPE_64, PCI_BASE_ADDRESS_SPACE_IO,
    PCI_BASE_ADDRESS_SPACE_MEMORY, PCI_CAP_ID_VNDR, PCI_CLASS_OTHERS, PCI_COMMAND,
    PCI_COMMAND_MASTER, PCI_INTERRUPT_PIN, PCI_PM_CTRL, PCI_PM_CTRL_STATE_MASK, PCI_PM_PMC,
    PCI_VENDOR_ID_REDHAT_QUMRANET, pci_set_long, pci_set_word,
};
use ruvm_hw_pci::{PciBus, PciDevice, PciDeviceInfo, PciDeviceOps};
use ruvm_mem::{AccessCtx, AccessSize, MemResult, MmioOps, RegionId};

use crate::net::VirtioNet;
use crate::virtio::{
    VIRTIO_CONFIG_S_ACKNOWLEDGE, VIRTIO_CONFIG_S_DRIVER, VIRTIO_CONFIG_S_DRIVER_OK,
    VIRTIO_F_BAD_FEATURE, VIRTIO_F_IOMMU_PLATFORM, VIRTIO_F_VERSION_1, VIRTIO_NO_VECTOR,
    VIRTIO_QUEUE_MAX, VirtIODevice, VirtioBackend, VirtioDeviceClass, VirtioTransport, feature,
};

mod vmstate;

pub use vmstate::{VirtioPciQueueVmState, VirtioPciVmState};

/// `TYPE_VIRTIO_PCI`, the abstract parent of every virtio PCI device.
pub const TYPE_VIRTIO_PCI: &str = "virtio-pci";
/// The generic virtio-rng PCI device.
pub const TYPE_VIRTIO_RNG_PCI: &str = "virtio-rng-pci";
/// The generic virtio-blk PCI device.
pub const TYPE_VIRTIO_BLK_PCI: &str = "virtio-blk-pci";
/// The generic virtio-serial PCI device, which carries virtio-console.
pub const TYPE_VIRTIO_SERIAL_PCI: &str = "virtio-serial-pci";
/// The generic virtio-net PCI device.
pub const TYPE_VIRTIO_NET_PCI: &str = "virtio-net-pci";
/// The generic virtio-balloon PCI device.
pub const TYPE_VIRTIO_BALLOON_PCI: &str = "virtio-balloon-pci";

/// `PCI_DEVICE_ID_VIRTIO_10_BASE`: modern-only devices use this plus the virtio device ID.
pub const PCI_DEVICE_ID_VIRTIO_10_BASE: u16 = 0x1040;
/// Transitional device ID of virtio-net.
pub const PCI_DEVICE_ID_VIRTIO_NET: u16 = 0x1000;
/// Transitional device ID of virtio-blk.
pub const PCI_DEVICE_ID_VIRTIO_BLOCK: u16 = 0x1001;
/// Transitional device ID of virtio-balloon.
pub const PCI_DEVICE_ID_VIRTIO_BALLOON: u16 = 0x1002;
/// Transitional device ID of virtio-serial (console).
pub const PCI_DEVICE_ID_VIRTIO_CONSOLE: u16 = 0x1003;
/// Transitional device ID of virtio-scsi.
pub const PCI_DEVICE_ID_VIRTIO_SCSI: u16 = 0x1004;
/// Transitional device ID of virtio-rng.
pub const PCI_DEVICE_ID_VIRTIO_RNG: u16 = 0x1005;
/// Transitional device ID of virtio-9p.
pub const PCI_DEVICE_ID_VIRTIO_9P: u16 = 0x1009;

/// `PCI_CLASS_STORAGE_SCSI`.
pub const PCI_CLASS_STORAGE_SCSI: u16 = 0x0100;
/// `PCI_CLASS_STORAGE_OTHER`.
pub const PCI_CLASS_STORAGE_OTHER: u16 = 0x0180;
/// `PCI_CLASS_NETWORK_ETHERNET`.
pub const PCI_CLASS_NETWORK_ETHERNET: u16 = 0x0200;
/// `PCI_BASE_CLASS_NETWORK`, what virtio-9p uses as its class.
pub const PCI_BASE_CLASS_NETWORK: u16 = 0x02;
/// `PCI_CLASS_DISPLAY_OTHER`.
pub const PCI_CLASS_DISPLAY_OTHER: u16 = 0x0380;
/// `PCI_CLASS_COMMUNICATION_OTHER`.
pub const PCI_CLASS_COMMUNICATION_OTHER: u16 = 0x0780;

/// Common configuration structure.
pub const VIRTIO_PCI_CAP_COMMON_CFG: u8 = 1;
/// Notifications.
pub const VIRTIO_PCI_CAP_NOTIFY_CFG: u8 = 2;
/// ISR status.
pub const VIRTIO_PCI_CAP_ISR_CFG: u8 = 3;
/// Device specific configuration.
pub const VIRTIO_PCI_CAP_DEVICE_CFG: u8 = 4;
/// PCI configuration access window.
pub const VIRTIO_PCI_CAP_PCI_CFG: u8 = 5;

/// `struct virtio_pci_cap` field offsets, relative to the capability.
pub const VIRTIO_PCI_CAP_LEN: usize = 2;
/// The `cfg_type` byte.
pub const VIRTIO_PCI_CAP_CFG_TYPE: usize = 3;
/// The `bar` byte.
pub const VIRTIO_PCI_CAP_BAR: usize = 4;
/// The `offset` word within the BAR.
pub const VIRTIO_PCI_CAP_OFFSET: usize = 8;
/// The `length` word.
pub const VIRTIO_PCI_CAP_LENGTH: usize = 12;
/// `notify_off_multiplier` in `struct virtio_pci_notify_cap`.
pub const VIRTIO_PCI_NOTIFY_CAP_MULT: usize = 16;
/// `pci_cfg_data` in `struct virtio_pci_cfg_cap`.
pub const VIRTIO_PCI_CFG_CAP_DATA: usize = 16;

/// Size of `struct virtio_pci_cap`.
const CAP_SIZE: u8 = 16;
/// Size of `struct virtio_pci_notify_cap` and `struct virtio_pci_cfg_cap`.
const CAP_SIZE_EXT: u8 = 20;

/// Common config: device feature word select.
pub const VIRTIO_PCI_COMMON_DFSELECT: u64 = 0;
/// Common config: device features, 32 bits at a time.
pub const VIRTIO_PCI_COMMON_DF: u64 = 4;
/// Common config: driver feature word select.
pub const VIRTIO_PCI_COMMON_GFSELECT: u64 = 8;
/// Common config: driver features, 32 bits at a time.
pub const VIRTIO_PCI_COMMON_GF: u64 = 12;
/// Common config: MSI-X vector for config changes.
pub const VIRTIO_PCI_COMMON_MSIX: u64 = 16;
/// Common config: number of queues.
pub const VIRTIO_PCI_COMMON_NUMQ: u64 = 18;
/// Common config: device status.
pub const VIRTIO_PCI_COMMON_STATUS: u64 = 20;
/// Common config: config generation.
pub const VIRTIO_PCI_COMMON_CFGGENERATION: u64 = 21;
/// Common config: queue select.
pub const VIRTIO_PCI_COMMON_Q_SELECT: u64 = 22;
/// Common config: size of the selected queue.
pub const VIRTIO_PCI_COMMON_Q_SIZE: u64 = 24;
/// Common config: MSI-X vector of the selected queue.
pub const VIRTIO_PCI_COMMON_Q_MSIX: u64 = 26;
/// Common config: the selected queue is enabled.
pub const VIRTIO_PCI_COMMON_Q_ENABLE: u64 = 28;
/// Common config: notify offset of the selected queue.
pub const VIRTIO_PCI_COMMON_Q_NOFF: u64 = 30;
/// Common config: descriptor table address, low half.
pub const VIRTIO_PCI_COMMON_Q_DESCLO: u64 = 32;
/// Common config: descriptor table address, high half.
pub const VIRTIO_PCI_COMMON_Q_DESCHI: u64 = 36;
/// Common config: available ring address, low half.
pub const VIRTIO_PCI_COMMON_Q_AVAILLO: u64 = 40;
/// Common config: available ring address, high half.
pub const VIRTIO_PCI_COMMON_Q_AVAILHI: u64 = 44;
/// Common config: used ring address, low half.
pub const VIRTIO_PCI_COMMON_Q_USEDLO: u64 = 48;
/// Common config: used ring address, high half.
pub const VIRTIO_PCI_COMMON_Q_USEDHI: u64 = 52;
/// Common config: notification data of the selected queue.
pub const VIRTIO_PCI_COMMON_Q_NDATA: u64 = 56;
/// Common config: reset the selected queue.
pub const VIRTIO_PCI_COMMON_Q_RESET: u64 = 58;

/// Legacy: device features, low 32 bits.
pub const VIRTIO_PCI_HOST_FEATURES: u64 = 0;
/// Legacy: driver features.
pub const VIRTIO_PCI_GUEST_FEATURES: u64 = 4;
/// Legacy: page frame number of the selected queue.
pub const VIRTIO_PCI_QUEUE_PFN: u64 = 8;
/// Legacy: size of the selected queue, read-only.
pub const VIRTIO_PCI_QUEUE_NUM: u64 = 12;
/// Legacy: queue select.
pub const VIRTIO_PCI_QUEUE_SEL: u64 = 14;
/// Legacy: queue notifier.
pub const VIRTIO_PCI_QUEUE_NOTIFY: u64 = 16;
/// Legacy: device status, one byte.
pub const VIRTIO_PCI_STATUS: u64 = 18;
/// Legacy: ISR status, one byte, cleared by reading.
pub const VIRTIO_PCI_ISR: u64 = 19;
/// Legacy with MSI-X enabled: config change vector.
pub const VIRTIO_MSI_CONFIG_VECTOR: u64 = 20;
/// Legacy with MSI-X enabled: vector of the selected queue.
pub const VIRTIO_MSI_QUEUE_VECTOR: u64 = 22;
/// `VIRTIO_PCI_QUEUE_ADDR_SHIFT`: the legacy PFN unit is 4 KiB.
pub const VIRTIO_PCI_QUEUE_ADDR_SHIFT: u32 = 12;
/// `VIRTIO_PCI_VRING_ALIGN`: legacy used ring alignment.
pub const VIRTIO_PCI_VRING_ALIGN: u32 = 4096;

/// Where the device config starts in the legacy window, `VIRTIO_PCI_CONFIG_OFF()`.
pub const fn virtio_pci_config_off(msix_enabled: bool) -> u64 {
    if msix_enabled { 24 } else { 20 }
}

/// BAR numbers of QEMU's default layout.
const LEGACY_IO_BAR: usize = 0;
const MSIX_BAR: u8 = 1;
const MODERN_IO_BAR: usize = 2;
const MODERN_MEM_BAR: usize = 4;

/// Offsets and sizes of the structures in the modern memory BAR.
const COMMON_OFFSET: u64 = 0x0;
const ISR_OFFSET: u64 = 0x1000;
const DEVICE_OFFSET: u64 = 0x2000;
const NOTIFY_OFFSET: u64 = 0x3000;
const REGION_SIZE: u64 = 0x1000;

/// `virtio_pci_optimal_num_queues()`: one queue per vCPU, limited by what MSI-X and virtio
/// can address once `fixed_queues` (control queues and the like) are taken.
pub fn virtio_pci_optimal_num_queues(fixed_queues: u32, cpus: u32) -> u32 {
    let msix_max = 0x7ff_u32.saturating_sub(fixed_queues);
    let virtio_max = (VIRTIO_QUEUE_MAX as u32).saturating_sub(fixed_queues);
    cpus.min(msix_max).min(virtio_max)
}

/// `virtio_legacy_allowed()`: device types that existed before virtio 1.0.
pub fn virtio_legacy_allowed(device_id: u16) -> bool {
    matches!(device_id, 1 | 2 | 3 | 4 | 5 | 7 | 8 | 9 | 11 | 12)
}

/// The transitional PCI device ID, the class and QEMU's type name stem of a virtio device ID.
fn device_table(id: u16) -> (Option<u16>, u16, Option<&'static str>) {
    match id {
        1 => (Some(PCI_DEVICE_ID_VIRTIO_NET), PCI_CLASS_NETWORK_ETHERNET, Some("virtio-net")),
        2 => (Some(PCI_DEVICE_ID_VIRTIO_BLOCK), PCI_CLASS_STORAGE_SCSI, Some("virtio-blk")),
        3 => (
            Some(PCI_DEVICE_ID_VIRTIO_CONSOLE),
            PCI_CLASS_COMMUNICATION_OTHER,
            Some("virtio-serial"),
        ),
        4 => (Some(PCI_DEVICE_ID_VIRTIO_RNG), PCI_CLASS_OTHERS, Some("virtio-rng")),
        5 => (Some(PCI_DEVICE_ID_VIRTIO_BALLOON), PCI_CLASS_OTHERS, Some("virtio-balloon")),
        8 => (Some(PCI_DEVICE_ID_VIRTIO_SCSI), PCI_CLASS_STORAGE_SCSI, Some("virtio-scsi")),
        9 => (Some(PCI_DEVICE_ID_VIRTIO_9P), PCI_BASE_CLASS_NETWORK, Some("virtio-9p")),
        19 => (None, PCI_CLASS_COMMUNICATION_OTHER, Some("vhost-vsock")),
        20 => (None, PCI_CLASS_OTHERS, Some("virtio-crypto")),
        26 => (None, PCI_CLASS_STORAGE_OTHER, Some("vhost-user-fs")),
        _ => (None, PCI_CLASS_OTHERS, None),
    }
}

/// `virtio_pci_get_class_id()`: the class code a device type reports.
pub fn virtio_pci_get_class_id(device_id: u16) -> u16 {
    device_table(device_id).1
}

/// `virtio_pci_get_trans_devid()`: the transitional PCI device ID, if the type has one.
pub fn virtio_pci_get_trans_devid(device_id: u16) -> Option<u16> {
    device_table(device_id).0
}

/// Which of the three QOM types registered by `virtio_pci_types_register()` a device is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VirtioPciVariant {
    /// `virtio-*-pci`: `disable-legacy` and `disable-modern` are left to the user.
    #[default]
    Generic,
    /// `virtio-*-pci-transitional`: legacy and modern both on.
    Transitional,
    /// `virtio-*-pci-non-transitional`: modern only.
    NonTransitional,
}

impl VirtioPciVariant {
    /// The type name for a generic base name such as `virtio-blk-pci`.
    pub fn type_name(self, base: &str) -> String {
        match self {
            VirtioPciVariant::Generic => base.to_string(),
            VirtioPciVariant::Transitional => format!("{base}-transitional"),
            VirtioPciVariant::NonTransitional => format!("{base}-non-transitional"),
        }
    }
}

/// The properties of `TYPE_VIRTIO_PCI` that this port understands.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VirtioPciProps {
    /// `disable-legacy`: `None` is auto, which disables legacy only behind a PCIe port.
    pub disable_legacy: Option<bool>,
    /// `disable-modern`.
    pub disable_modern: bool,
    /// `vectors`: the number of MSI-X vectors. `None` picks the device type's default, 0 means
    /// no MSI-X at all.
    pub vectors: Option<u32>,
    /// `class`: overrides the class code when not zero. virtio-serial only accepts a few.
    pub class_code: u16,
    /// `page-per-vq`: space queue notify addresses one page apart.
    pub page_per_vq: bool,
    /// `modern-pio-notify`: also offer an I/O port notify window in BAR2.
    pub modern_pio_notify: bool,
    /// The qdev id, used in error messages.
    pub id: Option<String>,
}

impl VirtioPciProps {
    /// The properties the given QOM type variant starts with.
    pub fn for_variant(variant: VirtioPciVariant) -> Self {
        match variant {
            VirtioPciVariant::Generic => Self::default(),
            VirtioPciVariant::Transitional => Self::transitional(),
            VirtioPciVariant::NonTransitional => Self::non_transitional(),
        }
    }

    /// `virtio-*-pci-transitional`.
    pub fn transitional() -> Self {
        VirtioPciProps { disable_legacy: Some(false), disable_modern: false, ..Self::default() }
    }

    /// `virtio-*-pci-non-transitional`.
    pub fn non_transitional() -> Self {
        VirtioPciProps { disable_legacy: Some(true), disable_modern: false, ..Self::default() }
    }
}

/// The `vectors` default of each device type's `realize` hook.
fn default_vectors(backend: &VirtioBackend) -> u32 {
    let vdev = backend.vdev();
    match vdev.device_id() {
        // virtio-net wants two per queue pair it may ever use, plus control and config. Only the
        // first pair exists until the driver turns multiqueue on, so this cannot count queues.
        1 => match backend.class().as_any().downcast_ref::<VirtioNet>() {
            Some(net) => 2 * u32::from(net.max_queue_pairs().max(1)) + 2,
            None => vdev.num_queues() as u32 + 1,
        },
        // virtio-blk and virtio-scsi use one per request queue plus one for config.
        2 | 8 | 26 => vdev.num_queues() as u32 + 1,
        19 => 3,
        _ => 2,
    }
}

/// `virtio_pci_queue_mem_mult()`.
fn queue_mem_mult(page_per_vq: bool) -> u64 {
    if page_per_vq { 0x1000 } else { 4 }
}

/// The transport side of interrupts, `virtio_pci_notify()`.
#[derive(Debug)]
struct PciNotify {
    dev: Arc<OnceLock<Weak<PciDevice>>>,
}

impl VirtioTransport for PciNotify {
    fn notify(&self, vector: u16, isr: u8) {
        let Some(dev) = self.dev.get().and_then(Weak::upgrade) else {
            return;
        };
        if dev.msix_enabled() {
            if vector != VIRTIO_NO_VECTOR && u32::from(vector) < dev.msix_nr_vectors_allocated() {
                dev.msix_notify(u32::from(vector));
            }
        } else {
            dev.set_irq(i32::from(isr & 1));
        }
    }
}

/// `VirtIOPCIQueue`: what a modern driver programs before enabling a queue.
#[derive(Clone, Copy, Debug, Default)]
struct PciQueue {
    num: u16,
    enabled: bool,
    reset: bool,
    desc: [u32; 2],
    avail: [u32; 2],
    used: [u32; 2],
}

fn join(halves: [u32; 2]) -> u64 {
    u64::from(halves[1]) << 32 | u64::from(halves[0])
}

#[derive(Debug)]
struct State {
    backend: VirtioBackend,
    dfselect: u32,
    gfselect: u32,
    guest_features: [u32; 2],
    vqs: Vec<PciQueue>,
}

/// The parts of `VirtIOPCIProxy` the region callbacks and config hooks share.
struct Inner {
    modern: bool,
    legacy: bool,
    notify_mult: u64,
    nvectors: AtomicU32,
    /// `config_cap`: offset of the config access capability, 0 without one.
    config_cap: AtomicU8,
    dev: Arc<OnceLock<Weak<PciDevice>>>,
    state: Mutex<State>,
    owner: Mutex<Option<ThreadId>>,
}

/// Holds the state lock and remembers which thread holds it, so a nested access from the same
/// thread can be refused instead of deadlocking.
struct StateGuard<'a> {
    state: MutexGuard<'a, State>,
    owner: &'a Mutex<Option<ThreadId>>,
}

impl Deref for StateGuard<'_> {
    type Target = State;
    fn deref(&self) -> &State {
        &self.state
    }
}

impl DerefMut for StateGuard<'_> {
    fn deref_mut(&mut self) -> &mut State {
        &mut self.state
    }
}

impl Drop for StateGuard<'_> {
    fn drop(&mut self) {
        *self.owner.lock().unwrap_or_else(PoisonError::into_inner) = None;
    }
}

/// The windows the proxy exposes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Region {
    Common,
    Isr,
    Device,
    Notify,
    NotifyPio,
    Legacy,
}

impl Inner {
    fn lock(&self) -> Option<StateGuard<'_>> {
        let me = thread::current().id();
        if *self.owner.lock().unwrap_or_else(PoisonError::into_inner) == Some(me) {
            return None;
        }
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        *self.owner.lock().unwrap_or_else(PoisonError::into_inner) = Some(me);
        Some(StateGuard { state, owner: &self.owner })
    }

    fn pci(&self) -> Option<Arc<PciDevice>> {
        self.dev.get().and_then(Weak::upgrade)
    }

    fn nvectors(&self) -> u32 {
        self.nvectors.load(Ordering::Relaxed)
    }

    /// `virtio_pci_reset()`: the device reset plus the proxy's own queue shadow state.
    fn proxy_reset(&self, dev: &PciDevice, st: &mut State) {
        st.backend.reset();
        dev.msix_unuse_all_vectors();
        st.guest_features = [0; 2];
        for q in &mut st.vqs {
            *q = PciQueue::default();
        }
    }

    /// Moves a config or queue vector from `old` to `val`, as the MSI-X vector registers of
    /// both layouts do. Returns the vector actually stored.
    fn switch_vector(&self, dev: &PciDevice, old: u16, val: u64) -> u16 {
        if old != VIRTIO_NO_VECTOR && dev.msix_present() {
            dev.msix_vector_unuse(u32::from(old));
        }
        // An out of range vector reads back as NO_VECTOR so the driver can see it failed.
        if val < u64::from(self.nvectors()) && dev.msix_present() {
            dev.msix_vector_use(val as u32);
            val as u16
        } else {
            VIRTIO_NO_VECTOR
        }
    }

    fn read(&self, dev: &PciDevice, st: &mut State, region: Region, addr: u64, size: u32) -> u64 {
        let val = match region {
            Region::Common => u64::from(self.common_read(st, addr)),
            Region::Isr => {
                let vdev = st.backend.vdev_mut();
                let isr = vdev.isr();
                vdev.clear_isr(u8::MAX);
                dev.deassert_intx();
                u64::from(isr)
            }
            Region::Device => match size {
                1 | 2 | 4 => u64::from(st.backend.config_read(addr, size)),
                _ => 0,
            },
            Region::Notify | Region::NotifyPio => 0,
            Region::Legacy => self.legacy_read(dev, st, addr, size),
        };
        val & size_mask(size)
    }

    fn write(
        &self,
        dev: &PciDevice,
        st: &mut State,
        region: Region,
        addr: u64,
        size: u32,
        val: u64,
    ) {
        match region {
            Region::Common => self.common_write(dev, st, addr, val),
            // The ISR is read-only.
            Region::Isr => {}
            Region::Device => {
                if let 1 | 2 | 4 = size {
                    st.backend.config_write(addr, size, val as u32);
                }
            }
            Region::Notify => {
                let queue = addr / self.notify_mult;
                if queue < VIRTIO_QUEUE_MAX as u64 {
                    st.backend.queue_notify(queue as u16);
                }
            }
            Region::NotifyPio => {
                let queue = val as u16;
                if usize::from(queue) < VIRTIO_QUEUE_MAX {
                    st.backend.queue_notify(queue);
                }
            }
            Region::Legacy => self.legacy_write(dev, st, addr, size, val),
        }
    }

    /// `virtio_pci_common_read()`.
    fn common_read(&self, st: &State, addr: u64) -> u32 {
        let vdev = st.backend.vdev();
        let sel = vdev.queue_sel();
        let q = &st.vqs[usize::from(sel)];
        match addr {
            VIRTIO_PCI_COMMON_DFSELECT => st.dfselect,
            VIRTIO_PCI_COMMON_DF => {
                if st.dfselect < 2 {
                    ((vdev.host_features() & !st.backend.class().legacy_features())
                        >> (32 * st.dfselect)) as u32
                } else {
                    0
                }
            }
            VIRTIO_PCI_COMMON_GFSELECT => st.gfselect,
            VIRTIO_PCI_COMMON_GF => {
                if st.gfselect < 2 {
                    st.guest_features[st.gfselect as usize]
                } else {
                    0
                }
            }
            VIRTIO_PCI_COMMON_MSIX => u32::from(vdev.config_vector()),
            VIRTIO_PCI_COMMON_NUMQ => (0..vdev.num_queues())
                .rev()
                .find(|&i| vdev.queue_num(i as u16) != 0)
                .map_or(0, |i| i as u32 + 1),
            VIRTIO_PCI_COMMON_STATUS => u32::from(vdev.status()),
            VIRTIO_PCI_COMMON_CFGGENERATION => vdev.generation(),
            VIRTIO_PCI_COMMON_Q_SELECT => u32::from(sel),
            VIRTIO_PCI_COMMON_Q_SIZE => u32::from(vdev.queue_num(sel)),
            VIRTIO_PCI_COMMON_Q_MSIX => {
                u32::from(vdev.queue(sel).map_or(VIRTIO_NO_VECTOR, |q| q.vector()))
            }
            VIRTIO_PCI_COMMON_Q_ENABLE => u32::from(q.enabled),
            // Queues are simply mapped in order.
            VIRTIO_PCI_COMMON_Q_NOFF => u32::from(sel),
            VIRTIO_PCI_COMMON_Q_DESCLO => q.desc[0],
            VIRTIO_PCI_COMMON_Q_DESCHI => q.desc[1],
            VIRTIO_PCI_COMMON_Q_AVAILLO => q.avail[0],
            VIRTIO_PCI_COMMON_Q_AVAILHI => q.avail[1],
            VIRTIO_PCI_COMMON_Q_USEDLO => q.used[0],
            VIRTIO_PCI_COMMON_Q_USEDHI => q.used[1],
            VIRTIO_PCI_COMMON_Q_RESET => u32::from(q.reset),
            _ => 0,
        }
    }

    /// `virtio_pci_common_write()`.
    fn common_write(&self, dev: &PciDevice, st: &mut State, addr: u64, val: u64) {
        let sel = st.backend.vdev().queue_sel();
        let idx = usize::from(sel);
        match addr {
            VIRTIO_PCI_COMMON_DFSELECT => st.dfselect = val as u32,
            VIRTIO_PCI_COMMON_GFSELECT => st.gfselect = val as u32,
            VIRTIO_PCI_COMMON_GF => {
                if st.gfselect < 2 {
                    st.guest_features[st.gfselect as usize] = val as u32;
                    let features = join(st.guest_features);
                    // Unsupported bits are dropped, which the driver sees when it reads back.
                    let _ = st.backend.set_features(features);
                }
            }
            VIRTIO_PCI_COMMON_MSIX => {
                let old = st.backend.vdev().config_vector();
                let v = self.switch_vector(dev, old, val);
                st.backend.vdev_mut().set_config_vector(v);
            }
            VIRTIO_PCI_COMMON_STATUS => {
                // A refused FEATURES_OK leaves the status alone, which is what the driver checks.
                let _ = st.backend.set_status(val as u8);
                if st.backend.vdev().status() == 0 {
                    self.proxy_reset(dev, st);
                }
            }
            VIRTIO_PCI_COMMON_Q_SELECT => {
                if val < VIRTIO_QUEUE_MAX as u64 {
                    st.backend.vdev_mut().set_queue_sel(val as u16);
                }
            }
            VIRTIO_PCI_COMMON_Q_SIZE => {
                st.vqs[idx].num = val as u16;
                st.backend.vdev_mut().set_queue_num(sel, u32::from(val as u16));
            }
            VIRTIO_PCI_COMMON_Q_MSIX => {
                let vdev = st.backend.vdev();
                let old = vdev.queue(sel).map_or(VIRTIO_NO_VECTOR, |q| q.vector());
                let v = self.switch_vector(dev, old, val);
                st.backend.vdev_mut().set_queue_vector(sel, v);
            }
            VIRTIO_PCI_COMMON_Q_ENABLE => {
                if val == 1 {
                    let q = st.vqs[idx];
                    let vdev = st.backend.vdev_mut();
                    vdev.set_queue_num(sel, u32::from(q.num));
                    vdev.set_queue_rings(sel, join(q.desc), join(q.avail), join(q.used));
                    st.vqs[idx].enabled = true;
                    st.vqs[idx].reset = false;
                } else {
                    st.backend.vdev_mut().error(&format!("wrong value for queue_enable {val:x}"));
                }
            }
            VIRTIO_PCI_COMMON_Q_DESCLO => st.vqs[idx].desc[0] = val as u32,
            VIRTIO_PCI_COMMON_Q_DESCHI => st.vqs[idx].desc[1] = val as u32,
            VIRTIO_PCI_COMMON_Q_AVAILLO => st.vqs[idx].avail[0] = val as u32,
            VIRTIO_PCI_COMMON_Q_AVAILHI => st.vqs[idx].avail[1] = val as u32,
            VIRTIO_PCI_COMMON_Q_USEDLO => st.vqs[idx].used[0] = val as u32,
            VIRTIO_PCI_COMMON_Q_USEDHI => st.vqs[idx].used[1] = val as u32,
            VIRTIO_PCI_COMMON_Q_RESET if val == 1 => {
                st.backend.vdev_mut().reset_queue(sel);
                st.vqs[idx].reset = false;
                st.vqs[idx].enabled = false;
            }
            // Read-only fields and holes ignore writes.
            _ => {}
        }
    }

    /// `virtio_pci_config_read()` and `virtio_ioport_read()`: the legacy window.
    fn legacy_read(&self, dev: &PciDevice, st: &mut State, addr: u64, size: u32) -> u64 {
        let config = virtio_pci_config_off(dev.msix_enabled());
        if addr >= config {
            return match size {
                1 | 2 | 4 => u64::from(st.backend.config_read(addr - config, size)),
                _ => 0,
            };
        }
        let vdev = st.backend.vdev_mut();
        let sel = vdev.queue_sel();
        let val: u32 = match addr {
            VIRTIO_PCI_HOST_FEATURES => vdev.host_features() as u32,
            VIRTIO_PCI_GUEST_FEATURES => vdev.guest_features() as u32,
            VIRTIO_PCI_QUEUE_PFN => (vdev.queue_addr(sel) >> VIRTIO_PCI_QUEUE_ADDR_SHIFT) as u32,
            VIRTIO_PCI_QUEUE_NUM => u32::from(vdev.queue_num(sel)),
            VIRTIO_PCI_QUEUE_SEL => u32::from(sel),
            VIRTIO_PCI_STATUS => u32::from(vdev.status()),
            VIRTIO_PCI_ISR => {
                // Reading the ISR also clears it.
                let isr = vdev.isr();
                vdev.clear_isr(u8::MAX);
                dev.deassert_intx();
                u32::from(isr)
            }
            VIRTIO_MSI_CONFIG_VECTOR => u32::from(vdev.config_vector()),
            VIRTIO_MSI_QUEUE_VECTOR => {
                u32::from(vdev.queue(sel).map_or(VIRTIO_NO_VECTOR, |q| q.vector()))
            }
            _ => u32::MAX,
        };
        u64::from(val)
    }

    /// `virtio_pci_config_write()` and `virtio_ioport_write()`.
    fn legacy_write(&self, dev: &PciDevice, st: &mut State, addr: u64, size: u32, val: u64) {
        let config = virtio_pci_config_off(dev.msix_enabled());
        if addr >= config {
            if let 1 | 2 | 4 = size {
                st.backend.config_write(addr - config, size, val as u32);
            }
            return;
        }
        let val = val as u32;
        let sel = st.backend.vdev().queue_sel();
        match addr {
            VIRTIO_PCI_GUEST_FEATURES => {
                // A driver that does not negotiate properly gets nothing.
                let val = if val & (1 << VIRTIO_F_BAD_FEATURE) != 0 { 0 } else { val };
                let _ = st.backend.set_features(u64::from(val));
            }
            VIRTIO_PCI_QUEUE_PFN => {
                let pa = u64::from(val) << VIRTIO_PCI_QUEUE_ADDR_SHIFT;
                if pa == 0 {
                    self.proxy_reset(dev, st);
                } else {
                    st.backend.vdev_mut().set_queue_addr(sel, pa);
                }
            }
            VIRTIO_PCI_QUEUE_SEL => {
                if (val as usize) < VIRTIO_QUEUE_MAX {
                    st.backend.vdev_mut().set_queue_sel(val as u16);
                }
            }
            VIRTIO_PCI_QUEUE_NOTIFY => {
                let n = val as u16;
                if usize::from(n) < VIRTIO_QUEUE_MAX && st.backend.vdev().queue_num(n) != 0 {
                    st.backend.queue_notify(n);
                }
            }
            VIRTIO_PCI_STATUS => {
                let _ = st.backend.set_status(val as u8);
                if st.backend.vdev().status() == 0 {
                    self.proxy_reset(dev, st);
                }
                // Linux before 2.6.34 drives the device without enabling bus mastering. QEMU
                // turns it on for the guest, and so do we.
                if val == u32::from(VIRTIO_CONFIG_S_ACKNOWLEDGE | VIRTIO_CONFIG_S_DRIVER) {
                    let cmd = dev.default_read_config(PCI_COMMAND as u32, 1);
                    dev.default_write_config(
                        PCI_COMMAND as u32,
                        cmd | u32::from(PCI_COMMAND_MASTER),
                        1,
                    );
                }
            }
            VIRTIO_MSI_CONFIG_VECTOR => {
                let old = st.backend.vdev().config_vector();
                let v = self.switch_vector(dev, old, u64::from(val));
                st.backend.vdev_mut().set_config_vector(v);
            }
            VIRTIO_MSI_QUEUE_VECTOR => {
                let old = st.backend.vdev().queue(sel).map_or(VIRTIO_NO_VECTOR, |q| q.vector());
                let v = self.switch_vector(dev, old, u64::from(val));
                st.backend.vdev_mut().set_queue_vector(sel, v);
            }
            // Read-only registers and holes ignore writes.
            _ => {}
        }
    }

    /// `virtio_address_space_lookup()`: which modern structure holds `len` bytes at `off`
    /// within BAR4, and the offset inside it.
    fn lookup(&self, off: u64, len: u64) -> Option<(Region, u64)> {
        let regions = [
            (Region::Common, COMMON_OFFSET, REGION_SIZE),
            (Region::Isr, ISR_OFFSET, REGION_SIZE),
            (Region::Device, DEVICE_OFFSET, REGION_SIZE),
            (Region::Notify, NOTIFY_OFFSET, self.notify_mult * VIRTIO_QUEUE_MAX as u64),
        ];
        regions
            .into_iter()
            .find(|&(_, start, size)| off >= start && off + len <= start + size)
            .map(|(r, start, _)| (r, off - start))
    }

    /// Reads the `offset` and `length` of the config access capability at `cap`.
    fn cfg_cap_target(dev: &PciDevice, cap: u32) -> Option<(u64, u32)> {
        let off = dev.default_read_config(cap + VIRTIO_PCI_CAP_OFFSET as u32, 4);
        let len = dev.default_read_config(cap + VIRTIO_PCI_CAP_LENGTH as u32, 4);
        if !matches!(len, 1 | 2 | 4) {
            return None;
        }
        // The address is under guest control, so align it rather than trusting it.
        Some((u64::from(off & !(len - 1)), len))
    }
}

fn size_mask(size: u32) -> u64 {
    if size >= 8 { u64::MAX } else { (1u64 << (8 * size)) - 1 }
}

/// Whether `[a, a + alen)` and `[b, b + blen)` overlap.
fn ranges_overlap(a: u32, alen: u32, b: u32, blen: u32) -> bool {
    a < b + blen && b < a + alen
}

/// One of the proxy's windows as an MMIO or I/O region.
struct RegionOps {
    inner: Arc<Inner>,
    region: Region,
}

impl fmt::Debug for RegionOps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegionOps").field("region", &self.region).finish_non_exhaustive()
    }
}

impl MmioOps for RegionOps {
    fn read(&self, _cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        let Some(dev) = self.inner.pci() else {
            return Ok(size.mask());
        };
        let Some(mut st) = self.inner.lock() else {
            return Ok(0);
        };
        Ok(self.inner.read(&dev, &mut st, self.region, offset, size.bytes()))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, size: AccessSize, value: u64) -> MemResult<()> {
        let Some(dev) = self.inner.pci() else {
            return Ok(());
        };
        if let Some(mut st) = self.inner.lock() {
            self.inner.write(&dev, &mut st, self.region, offset, size.bytes(), value);
        }
        Ok(())
    }
}

/// The config space and reset hooks, `virtio_read_config()`, `virtio_write_config()` and
/// `virtio_pci_reset()`.
struct PciOps {
    inner: Arc<Inner>,
}

impl PciDeviceOps for PciOps {
    fn config_read(&self, dev: &PciDevice, addr: u32, len: u32) -> u32 {
        let cap = u32::from(self.inner.config_cap.load(Ordering::Relaxed));
        if cap != 0 && ranges_overlap(addr, len, cap + VIRTIO_PCI_CFG_CAP_DATA as u32, 4) {
            if let Some((off, caplen)) = Inner::cfg_cap_target(dev, cap) {
                if let Some((region, raddr)) = self.inner.lookup(off, u64::from(caplen)) {
                    if let Some(mut st) = self.inner.lock() {
                        let val = self.inner.read(dev, &mut st, region, raddr, caplen) as u32;
                        drop(st);
                        let data = (cap as usize) + VIRTIO_PCI_CFG_CAP_DATA;
                        dev.with_config(|c| {
                            c.config[data..data + caplen as usize]
                                .copy_from_slice(&val.to_le_bytes()[..caplen as usize]);
                        });
                    }
                }
            }
        }
        dev.default_read_config(addr, len)
    }

    fn config_write(&self, dev: &PciDevice, addr: u32, val: u32, len: u32) {
        dev.default_write_config(addr, val, len);

        let covers_command = addr <= PCI_COMMAND as u32 && (PCI_COMMAND as u32) < addr + len;
        let cap = u32::from(self.inner.config_cap.load(Ordering::Relaxed));
        let hits_cfg_data =
            cap != 0 && ranges_overlap(addr, len, cap + VIRTIO_PCI_CFG_CAP_DATA as u32, 4);
        if !covers_command && !hits_cfg_data {
            return;
        }
        let Some(mut st) = self.inner.lock() else {
            return;
        };

        if covers_command {
            let cmd = dev.default_read_config(PCI_COMMAND as u32, 1);
            if cmd & u32::from(PCI_COMMAND_MASTER) == 0 {
                st.backend.vdev_mut().set_disabled(true);
                let status = st.backend.vdev().status() & !VIRTIO_CONFIG_S_DRIVER_OK;
                let _ = st.backend.set_status(status);
            } else {
                st.backend.vdev_mut().set_disabled(false);
            }
        }

        if hits_cfg_data {
            if let Some((off, caplen)) = Inner::cfg_cap_target(dev, cap) {
                if let Some((region, raddr)) = self.inner.lookup(off, u64::from(caplen)) {
                    let data = dev.default_read_config(cap + VIRTIO_PCI_CFG_CAP_DATA as u32, 4);
                    let data = u64::from(data) & size_mask(caplen);
                    self.inner.write(dev, &mut st, region, raddr, caplen, data);
                }
            }
        }
    }

    fn reset(&self, dev: &PciDevice) {
        if let Some(mut st) = self.inner.lock() {
            self.inner.proxy_reset(dev, &mut st);
        }
    }
}

/// A virtio PCI function, `VirtIOPCIProxy` with its device plugged in. A clone is another
/// handle to the same function.
#[derive(Clone)]
pub struct VirtioPci {
    pci: Arc<PciDevice>,
    inner: Arc<Inner>,
}

impl fmt::Debug for VirtioPci {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VirtioPci")
            .field("pci", &self.pci.name())
            .field("modern", &self.inner.modern)
            .field("legacy", &self.inner.legacy)
            .field("nvectors", &self.inner.nvectors())
            .finish_non_exhaustive()
    }
}

/// Is the bus a PCI Express port, the `pcie_port` test of `virtio_pci_realize()`? The root bus
/// never is, and a secondary bus is express when the bridge above it is.
fn is_pcie_port(bus: &PciBus) -> bool {
    !bus.is_root() && bus.parent_dev().is_some_and(|d| d.is_express())
}

fn mem_err(e: impl fmt::Display) -> Error {
    Error::generic(e.to_string())
}

/// `virtio_pci_add_mem_cap()`: adds a vendor capability describing `length` bytes at `offset`
/// in `bar`. `mult` makes it a notify capability.
fn add_mem_cap(
    dev: &PciDevice,
    cfg_type: u8,
    bar: usize,
    region: (u64, u64),
    mult: Option<u32>,
) -> Result<u8> {
    let cap_len = if mult.is_some() { CAP_SIZE_EXT } else { CAP_SIZE };
    let pos = dev.add_capability(PCI_CAP_ID_VNDR, 0, cap_len)?;
    let p = usize::from(pos);
    dev.with_config(|c| {
        c.config[p + VIRTIO_PCI_CAP_LEN] = cap_len;
        c.config[p + VIRTIO_PCI_CAP_CFG_TYPE] = cfg_type;
        c.config[p + VIRTIO_PCI_CAP_BAR] = bar as u8;
        pci_set_long(c.config, p + VIRTIO_PCI_CAP_OFFSET, region.0 as u32);
        pci_set_long(c.config, p + VIRTIO_PCI_CAP_LENGTH, region.1 as u32);
        if let Some(m) = mult {
            pci_set_long(c.config, p + VIRTIO_PCI_NOTIFY_CAP_MULT, m);
        }
    });
    Ok(pos)
}

impl VirtioPci {
    /// Registers a virtio PCI function at `devfn` (or the first free slot) and plugs `backend`
    /// into it. This is `virtio_pci_realize()` followed by `virtio_pci_device_plugged()`.
    ///
    /// The bus's root must have an MSI handler for MSI-X to be offered. Without one the device
    /// falls back to INTx, as QEMU does when MSI is not supported.
    pub fn new(
        bus: &Arc<PciBus>,
        devfn: Option<u8>,
        mut backend: VirtioBackend,
        props: &VirtioPciProps,
    ) -> Result<VirtioPci> {
        let pcie_port = is_pcie_port(bus);
        let legacy = !props.disable_legacy.unwrap_or(pcie_port);
        let mut modern = !props.disable_modern;
        if !modern && !legacy {
            return Err(Error::generic(
                "device cannot work as neither modern nor legacy mode is enabled",
            ));
        }
        let notify_mult = queue_mem_mult(props.page_per_vq);

        // virtio_pci_pre_plugged().
        let slot = Arc::new(OnceLock::new());
        let mut transport_features = feature(VIRTIO_F_BAD_FEATURE);
        if modern {
            transport_features |= feature(VIRTIO_F_VERSION_1);
        }
        backend.plug(Arc::new(PciNotify { dev: Arc::clone(&slot) }), transport_features)?;

        let vdev = backend.vdev();
        let id = vdev.device_id();
        if !vdev.host_has_feature(VIRTIO_F_VERSION_1) {
            modern = false;
            if !legacy {
                return Err(Error::generic(
                    "Device doesn't support modern mode, and legacy mode is disabled",
                ));
            }
        }
        if legacy {
            if !virtio_legacy_allowed(id) {
                return Err(Error::generic(format!(
                    "device is modern-only, use disable-legacy=on ({})",
                    vdev.name()
                )));
            }
            if vdev.host_has_feature(VIRTIO_F_IOMMU_PLATFORM) {
                return Err(Error::generic(
                    "VIRTIO_F_IOMMU_PLATFORM was supported by neither legacy nor transitional \
                     device",
                ));
            }
        }

        let (trans_devid, class, stem) = device_table(id);
        let mut class_code = props.class_code;
        if id == 3
            && !matches!(
                class_code,
                PCI_CLASS_COMMUNICATION_OTHER | PCI_CLASS_DISPLAY_OTHER | PCI_CLASS_OTHERS
            )
        {
            class_code = PCI_CLASS_COMMUNICATION_OTHER;
        }
        let modern_devid = PCI_DEVICE_ID_VIRTIO_10_BASE + id;
        let info = PciDeviceInfo {
            name: stem.map_or_else(|| format!("{}-pci", vdev.name()), |s| format!("{s}-pci")),
            id: props.id.clone(),
            vendor_id: PCI_VENDOR_ID_REDHAT_QUMRANET,
            device_id: if legacy { trans_devid.unwrap_or(modern_devid) } else { modern_devid },
            revision: if legacy { 0 } else { 1 },
            class_id: if class_code != 0 { class_code } else { class },
            // A legacy driver finds the device type in the subsystem ID.
            subsystem_vendor_id: if legacy { PCI_VENDOR_ID_REDHAT_QUMRANET } else { 0 },
            subsystem_id: if legacy { id } else { 0 },
            express: pcie_port && modern,
            ..PciDeviceInfo::default()
        };
        let nvectors = props.vectors.unwrap_or_else(|| default_vectors(&backend));
        let config_len = vdev.config_len() as u64;
        let nqueues = vdev.num_queues();

        // Legacy drivers expect the used ring 4 KiB aligned.
        for n in 0..nqueues {
            backend.vdev_mut().set_queue_align(n as u16, VIRTIO_PCI_VRING_ALIGN);
        }

        let dev = bus.register_device(&info, devfn)?;
        let _ = slot.set(Arc::downgrade(&dev));
        dev.with_config(|c| c.config[PCI_INTERRUPT_PIN] = 1);

        if pcie_port && modern {
            let pos = usize::from(dev.pm_init(0)?);
            dev.with_config(|c| {
                // Revision 1.2 of the PCI Power Management Interface Specification.
                pci_set_word(c.config, pos + PCI_PM_PMC, 0x3);
                pci_set_word(c.wmask, pos + PCI_PM_CTRL, PCI_PM_CTRL_STATE_MASK);
            });
        }

        let inner = Arc::new(Inner {
            modern,
            legacy,
            notify_mult,
            nvectors: AtomicU32::new(nvectors),
            config_cap: AtomicU8::new(0),
            dev: slot,
            state: Mutex::new(State {
                backend,
                dfselect: 0,
                gfselect: 0,
                guest_features: [0; 2],
                vqs: vec![PciQueue::default(); VIRTIO_QUEUE_MAX],
            }),
            owner: Mutex::new(None),
        });

        let mem = Arc::clone(bus.memory());
        let region = |name: &str, size: u64, r: Region| -> Result<RegionId> {
            let ops = Arc::new(RegionOps { inner: Arc::clone(&inner), region: r });
            mem.new_io(name, u128::from(size), ops).map_err(mem_err)
        };

        if modern {
            let notify_size = notify_mult * VIRTIO_QUEUE_MAX as u64;
            let bar_size = (NOTIFY_OFFSET + notify_size).next_power_of_two();
            let bar = mem.new_container("virtio-pci", u128::from(bar_size)).map_err(mem_err)?;
            let parts = [
                ("virtio-pci-common", COMMON_OFFSET, REGION_SIZE, Region::Common),
                ("virtio-pci-isr", ISR_OFFSET, REGION_SIZE, Region::Isr),
                ("virtio-pci-device", DEVICE_OFFSET, REGION_SIZE, Region::Device),
                ("virtio-pci-notify", NOTIFY_OFFSET, notify_size, Region::Notify),
            ];
            for (name, off, size, r) in parts {
                let id = region(name, size, r)?;
                mem.add_subregion(bar, off, id).map_err(mem_err)?;
            }
            add_mem_cap(
                &dev,
                VIRTIO_PCI_CAP_COMMON_CFG,
                MODERN_MEM_BAR,
                (COMMON_OFFSET, REGION_SIZE),
                None,
            )?;
            add_mem_cap(
                &dev,
                VIRTIO_PCI_CAP_ISR_CFG,
                MODERN_MEM_BAR,
                (ISR_OFFSET, REGION_SIZE),
                None,
            )?;
            add_mem_cap(
                &dev,
                VIRTIO_PCI_CAP_DEVICE_CFG,
                MODERN_MEM_BAR,
                (DEVICE_OFFSET, REGION_SIZE),
                None,
            )?;
            add_mem_cap(
                &dev,
                VIRTIO_PCI_CAP_NOTIFY_CFG,
                MODERN_MEM_BAR,
                (NOTIFY_OFFSET, notify_size),
                Some(notify_mult as u32),
            )?;
            if props.modern_pio_notify {
                let pio = region("virtio-pci-notify-pio", 4, Region::NotifyPio)?;
                add_mem_cap(&dev, VIRTIO_PCI_CAP_NOTIFY_CFG, MODERN_IO_BAR, (0, 4), Some(0))?;
                dev.register_bar(MODERN_IO_BAR, PCI_BASE_ADDRESS_SPACE_IO, pio);
            }
            dev.register_bar(
                MODERN_MEM_BAR,
                PCI_BASE_ADDRESS_SPACE_MEMORY
                    | PCI_BASE_ADDRESS_MEM_PREFETCH
                    | PCI_BASE_ADDRESS_MEM_TYPE_64,
                bar,
            );

            let cap = add_mem_cap(&dev, VIRTIO_PCI_CAP_PCI_CFG, 0, (0, 0), Some(0))?;
            let p = usize::from(cap);
            dev.with_config(|c| {
                c.wmask[p + VIRTIO_PCI_CAP_BAR] = 0xff;
                pci_set_long(c.wmask, p + VIRTIO_PCI_CAP_OFFSET, u32::MAX);
                pci_set_long(c.wmask, p + VIRTIO_PCI_CAP_LENGTH, u32::MAX);
                pci_set_long(c.wmask, p + VIRTIO_PCI_CFG_CAP_DATA, u32::MAX);
            });
            inner.config_cap.store(cap, Ordering::Relaxed);
        }

        if nvectors != 0 {
            if let Err(e) = dev.msix_init_exclusive_bar(nvectors, MSIX_BAR) {
                // Not having MSI at all is normal and not worth a warning.
                if bus.msi_nonbroken() {
                    warn_report(&format!("unable to init msix vectors to {nvectors}: {e}"));
                }
                inner.nvectors.store(0, Ordering::Relaxed);
            }
        }

        if legacy {
            let size = (virtio_pci_config_off(dev.msix_present()) + config_len).next_power_of_two();
            let io = region("virtio-pci", size, Region::Legacy)?;
            dev.register_bar(LEGACY_IO_BAR, PCI_BASE_ADDRESS_SPACE_IO, io);
        }

        dev.set_ops(Arc::new(PciOps { inner: Arc::clone(&inner) }));
        Ok(VirtioPci { pci: dev, inner })
    }

    /// The PCI function.
    pub fn pci_dev(&self) -> &Arc<PciDevice> {
        &self.pci
    }

    /// Whether the virtio 1.0 interface (BAR4 and the vendor capabilities) is offered.
    pub fn is_modern(&self) -> bool {
        self.inner.modern
    }

    /// Whether the legacy I/O BAR is offered, making the device transitional.
    pub fn is_legacy(&self) -> bool {
        self.inner.legacy
    }

    /// The number of MSI-X vectors, 0 when the device uses INTx only.
    pub fn nvectors(&self) -> u32 {
        self.inner.nvectors()
    }

    /// Runs `f` on the plugged device. Returns `None` if called from inside the device's own
    /// processing.
    pub fn with_backend<R>(&self, f: impl FnOnce(&mut VirtioBackend) -> R) -> Option<R> {
        let mut st = self.inner.lock()?;
        Some(f(&mut st.backend))
    }

    /// Runs `f` on the plugged device model as its concrete type `D`, together with the core
    /// state. Returns `None` if it is not a `D`.
    pub fn with_device<D: VirtioDeviceClass, R>(
        &self,
        f: impl FnOnce(&mut VirtIODevice, &mut D) -> R,
    ) -> Option<R> {
        let mut st = self.inner.lock()?;
        let (vdev, dev) = st.backend.downcast_mut::<D>()?;
        Some(f(vdev, dev))
    }

    /// A device reset: `virtio_pci_reset()` followed by the generic PCI function reset.
    pub fn reset(&self) {
        self.pci.reset();
    }
}
