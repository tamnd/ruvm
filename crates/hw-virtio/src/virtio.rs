// SPDX-License-Identifier: GPL-2.0-or-later

//! The virtio device core, a port of `hw/virtio/virtio.c`.
//!
//! [`VirtIODevice`] holds the state every virtio device has no matter which transport carries it:
//! device status, host and guest feature bits, the interrupt status byte, the config space
//! generation counter and the virtqueues. A device model implements [`VirtioDeviceClass`] and the
//! two are held together by [`VirtioBackend`], which is where the functions that need both live
//! (`virtio_set_status()`, `virtio_set_features()`, `virtio_reset()`, config space access and
//! queue notification). Transports drive a [`VirtioBackend`] and receive interrupts through
//! [`VirtioTransport`].
//!
//! Descriptor rings are handled by `ruvm-virtio-queue`. A [`VirtQueue`] keeps the transport
//! visible queue registers (size, alignment, ring addresses, vector) and builds a
//! [`SplitQueue`] or [`PackedQueue`] from them once they describe a ring.
//!
//! Differences from QEMU:
//!
//! - Features are a `u64`. QEMU 11 widened them to 128 bits, but no transport or device ported
//!   here uses a bit above 63.
//! - Only little endian rings and config space are supported, which is what every modern device
//!   uses and what legacy devices use on little endian targets.
//! - Queue memory errors and malformed chains found by `ruvm-virtio-queue` put the device in the
//!   broken state through [`VirtIODevice::error`], like `virtio_error()` does for the checks in
//!   `virtqueue_pop()`. The messages are not word for word the same.
//! - The in use counter counts chains, not descriptors, for packed rings as well.
//!
//! Not ported: VMState, trace points, QOM registration, ioeventfd and irqfd, host notifiers,
//! vhost, the IOMMU and memory listener integration, per-queue vectors beyond storing them,
//! `VIRTIO_F_NOTIFICATION_DATA` and `VIRTIO_F_IN_ORDER` handling, and queue reset through the
//! transport (the per-queue reset bit is still offered by default as QEMU does, but the MMIO
//! transport has no register for it).

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use ruvm_base::{Error, Result, error_report, warn_report};
use ruvm_virtio_queue::{
    DescriptorChain, GuestMemory, PackedQueue, QueueError, RingAddresses, SplitQueue,
    VIRTIO_F_EVENT_IDX, VIRTIO_F_INDIRECT_DESC, VIRTIO_F_RING_PACKED,
};

/// Guest memory shared between the device, its queues and whoever set it up.
pub type SharedGuestMemory = Arc<dyn GuestMemory + Send + Sync>;

/// `VIRTIO_CONFIG_S_ACKNOWLEDGE`: the guest has noticed the device.
pub const VIRTIO_CONFIG_S_ACKNOWLEDGE: u8 = 1;
/// `VIRTIO_CONFIG_S_DRIVER`: the guest has a driver for the device.
pub const VIRTIO_CONFIG_S_DRIVER: u8 = 2;
/// `VIRTIO_CONFIG_S_DRIVER_OK`: the driver is ready to drive the device.
pub const VIRTIO_CONFIG_S_DRIVER_OK: u8 = 4;
/// `VIRTIO_CONFIG_S_FEATURES_OK`: feature negotiation is complete.
pub const VIRTIO_CONFIG_S_FEATURES_OK: u8 = 8;
/// `VIRTIO_CONFIG_S_NEEDS_RESET`: the device hit an error and needs a reset.
pub const VIRTIO_CONFIG_S_NEEDS_RESET: u8 = 0x40;
/// `VIRTIO_CONFIG_S_FAILED`: the driver gave up on the device.
pub const VIRTIO_CONFIG_S_FAILED: u8 = 0x80;

/// `VIRTIO_F_NOTIFY_ON_EMPTY` (legacy only).
pub const VIRTIO_F_NOTIFY_ON_EMPTY: u32 = 24;
/// `VIRTIO_F_ANY_LAYOUT` (legacy only).
pub const VIRTIO_F_ANY_LAYOUT: u32 = 27;
/// `VIRTIO_F_BAD_FEATURE`: a legacy driver acking this is broken.
pub const VIRTIO_F_BAD_FEATURE: u32 = 30;
/// `VIRTIO_F_VERSION_1`: the device follows the virtio 1.0 specification.
pub const VIRTIO_F_VERSION_1: u32 = 32;
/// `VIRTIO_F_IOMMU_PLATFORM`, also known as `VIRTIO_F_ACCESS_PLATFORM`.
pub const VIRTIO_F_IOMMU_PLATFORM: u32 = 33;
/// `VIRTIO_F_IN_ORDER`.
pub const VIRTIO_F_IN_ORDER: u32 = 35;
/// `VIRTIO_F_NOTIFICATION_DATA`.
pub const VIRTIO_F_NOTIFICATION_DATA: u32 = 38;
/// `VIRTIO_F_RING_RESET`.
pub const VIRTIO_F_RING_RESET: u32 = 40;

pub use ruvm_virtio_queue::{
    VIRTIO_F_EVENT_IDX as VIRTIO_RING_F_EVENT_IDX,
    VIRTIO_F_INDIRECT_DESC as VIRTIO_RING_F_INDIRECT_DESC,
};

/// `VIRTIO_QUEUE_MAX`: how many queues a device can have.
pub const VIRTIO_QUEUE_MAX: usize = 1024;
/// `VIRTQUEUE_MAX_SIZE`: the largest queue size a device may offer.
pub const VIRTQUEUE_MAX_SIZE: u16 = 1024;
/// `VIRTIO_NO_VECTOR`.
pub const VIRTIO_NO_VECTOR: u16 = 0xffff;

/// The single bit mask of feature `bit`.
pub const fn feature(bit: u32) -> u64 {
    1 << bit
}

/// `virtio_has_feature()`: whether `bit` is set in `features`.
pub const fn has_feature(features: u64, bit: u32) -> bool {
    features & feature(bit) != 0
}

/// `VIRTIO_LEGACY_FEATURES`: the bits a modern transport hides from the device feature
/// registers.
pub const VIRTIO_LEGACY_FEATURES: u64 = feature(VIRTIO_F_BAD_FEATURE)
    | feature(VIRTIO_F_NOTIFY_ON_EMPTY)
    | feature(VIRTIO_F_ANY_LAYOUT);

/// The host features every device starts with, `DEFINE_VIRTIO_COMMON_FEATURES()` with its
/// defaults: indirect descriptors, event index, notify on empty, any layout and queue reset.
pub const VIRTIO_COMMON_FEATURES: u64 = feature(VIRTIO_F_INDIRECT_DESC)
    | feature(VIRTIO_F_EVENT_IDX)
    | feature(VIRTIO_F_NOTIFY_ON_EMPTY)
    | feature(VIRTIO_F_ANY_LAYOUT)
    | feature(VIRTIO_F_RING_RESET);

/// What a device uses to reach the guest, `VirtioBusClass::notify`.
///
/// The transport decides how an interrupt is delivered. The MMIO transport ignores `vector` and
/// sets its interrupt line to `isr != 0`.
pub trait VirtioTransport: Send + Sync + fmt::Debug {
    /// The device changed its interrupt status. `vector` is the queue or config vector the
    /// change is about, or [`VIRTIO_NO_VECTOR`], and `isr` is the interrupt status byte after
    /// the change.
    fn notify(&self, vector: u16, isr: u8);
}

/// The ring a queue currently runs on.
#[derive(Clone, Debug)]
enum Ring {
    Split(SplitQueue),
    Packed(PackedQueue),
}

impl Ring {
    fn set_flags(&mut self, event_idx: bool, indirect: bool) {
        match self {
            Ring::Split(q) => {
                q.set_event_idx(event_idx);
                q.set_indirect_desc(indirect);
            }
            Ring::Packed(q) => {
                q.set_event_idx(event_idx);
                q.set_indirect_desc(indirect);
            }
        }
    }

    fn pop<M: GuestMemory + ?Sized>(
        &mut self,
        mem: &M,
    ) -> std::result::Result<Option<DescriptorChain>, QueueError> {
        match self {
            Ring::Split(q) => q.pop(mem),
            Ring::Packed(q) => q.pop(mem),
        }
    }

    fn add_used<M: GuestMemory + ?Sized>(
        &mut self,
        mem: &M,
        chain: &DescriptorChain,
        len: u32,
    ) -> std::result::Result<(), QueueError> {
        match self {
            Ring::Split(q) => q.add_used(mem, chain.head(), len),
            Ring::Packed(q) => q.add_used(mem, chain.head(), len, chain.ring_slots()),
        }
    }

    fn has_available<M: GuestMemory + ?Sized>(
        &self,
        mem: &M,
    ) -> std::result::Result<bool, QueueError> {
        match self {
            Ring::Split(q) => q.has_available(mem),
            Ring::Packed(q) => q.has_available(mem),
        }
    }

    fn needs_notification<M: GuestMemory + ?Sized>(
        &mut self,
        mem: &M,
    ) -> std::result::Result<bool, QueueError> {
        match self {
            Ring::Split(q) => q.needs_notification(mem),
            Ring::Packed(q) => q.needs_notification(mem),
        }
    }

    fn set_notification<M: GuestMemory + ?Sized>(
        &mut self,
        mem: &M,
        enable: bool,
    ) -> std::result::Result<(), QueueError> {
        match (self, enable) {
            (Ring::Split(q), true) => q.enable_notification(mem).map(|_| ()),
            (Ring::Split(q), false) => q.disable_notification(mem),
            (Ring::Packed(q), true) => q.enable_notification(mem).map(|_| ()),
            (Ring::Packed(q), false) => q.disable_notification(mem),
        }
    }
}

/// One virtqueue as the transport sees it, `VirtQueue` with its `VRing`.
#[derive(Clone, Debug)]
pub struct VirtQueue {
    num: u16,
    num_default: u16,
    align: u32,
    desc: u64,
    avail: u64,
    used: u64,
    vector: u16,
    inuse: u32,
    notification: bool,
    ring: Option<Ring>,
    ring_error: Option<String>,
}

impl VirtQueue {
    fn new(size: u16) -> Self {
        VirtQueue {
            num: size,
            num_default: size,
            align: 0,
            desc: 0,
            avail: 0,
            used: 0,
            vector: VIRTIO_NO_VECTOR,
            inuse: 0,
            notification: true,
            ring: None,
            ring_error: None,
        }
    }

    /// The current queue size.
    pub fn num(&self) -> u16 {
        self.num
    }

    /// The largest size the device offers, the size it was added with.
    pub fn num_default(&self) -> u16 {
        self.num_default
    }

    /// The legacy ring alignment.
    pub fn align(&self) -> u32 {
        self.align
    }

    /// Descriptor table address.
    pub fn desc(&self) -> u64 {
        self.desc
    }

    /// Available ring (driver area) address.
    pub fn avail(&self) -> u64 {
        self.avail
    }

    /// Used ring (device area) address.
    pub fn used(&self) -> u64 {
        self.used
    }

    /// The MSI vector of the queue.
    pub fn vector(&self) -> u16 {
        self.vector
    }

    /// How many chains the device has taken and not returned yet.
    pub fn inuse(&self) -> u32 {
        self.inuse
    }

    /// `__virtio_queue_reset()`. The alignment survives, as in QEMU.
    fn reset(&mut self) {
        self.desc = 0;
        self.avail = 0;
        self.used = 0;
        self.vector = VIRTIO_NO_VECTOR;
        self.inuse = 0;
        self.notification = true;
        self.num = self.num_default;
        self.ring = None;
        self.ring_error = None;
    }

    /// Builds the ring from the registers, the moral equivalent of
    /// `virtio_init_region_cache()`. Indices carry over when a split ring is rebuilt as a split
    /// ring, since QEMU does not reset them when addresses change either.
    fn rebuild(&mut self, guest_features: u64) {
        self.ring_error = None;
        if self.num == 0 || self.desc == 0 {
            self.ring = None;
            return;
        }
        let addrs = RingAddresses {
            desc_table: self.desc,
            driver_area: self.avail,
            device_area: self.used,
        };
        let event_idx = has_feature(guest_features, VIRTIO_F_EVENT_IDX);
        let indirect = has_feature(guest_features, VIRTIO_F_INDIRECT_DESC);
        let built = if has_feature(guest_features, VIRTIO_F_RING_PACKED) {
            PackedQueue::new(self.num, addrs).map(Ring::Packed)
        } else {
            SplitQueue::new(self.num, addrs).map(|mut q| {
                if let Some(Ring::Split(old)) = &self.ring {
                    q.set_next_avail(old.next_avail());
                    q.set_next_used(old.next_used());
                }
                Ring::Split(q)
            })
        };
        match built {
            Ok(mut ring) => {
                ring.set_flags(event_idx, indirect);
                self.ring = Some(ring);
            }
            Err(e) => {
                self.ring = None;
                self.ring_error = Some(e.to_string());
            }
        }
    }

    /// `virtio_queue_update_rings()`: derives the legacy ring layout from the descriptor table
    /// address, the size and the alignment.
    fn update_rings(&mut self, guest_features: u64) {
        if self.num == 0 || self.desc == 0 || self.align == 0 {
            return;
        }
        let num = u64::from(self.num);
        self.avail = self.desc + num * 16;
        let align = u64::from(self.align);
        self.used = (self.avail + 4 + 2 * num).div_ceil(align) * align;
        self.rebuild(guest_features);
    }
}

/// The state shared by all virtio devices, `VirtIODevice`.
pub struct VirtIODevice {
    name: String,
    device_id: u16,
    config: Vec<u8>,
    host_features: u64,
    guest_features: u64,
    status: u8,
    isr: u8,
    queue_sel: u16,
    generation: u32,
    broken: bool,
    started: bool,
    start_on_kick: bool,
    disabled: bool,
    config_vector: u16,
    vqs: Vec<VirtQueue>,
    mem: SharedGuestMemory,
    transport: Option<Arc<dyn VirtioTransport>>,
}

impl fmt::Debug for VirtIODevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VirtIODevice")
            .field("name", &self.name)
            .field("device_id", &self.device_id)
            .field("host_features", &format_args!("{:#x}", self.host_features))
            .field("guest_features", &format_args!("{:#x}", self.guest_features))
            .field("status", &self.status)
            .field("isr", &self.isr)
            .field("broken", &self.broken)
            .field("queues", &self.vqs.len())
            .finish_non_exhaustive()
    }
}

impl VirtIODevice {
    /// A device with no identity, no config space and no queues that reaches the guest through
    /// `mem`. The device model fills it in from [`VirtioDeviceClass::realize`].
    pub fn new(mem: SharedGuestMemory) -> Self {
        VirtIODevice {
            name: String::new(),
            device_id: 0,
            config: Vec::new(),
            host_features: VIRTIO_COMMON_FEATURES,
            guest_features: 0,
            status: 0,
            isr: 0,
            queue_sel: 0,
            generation: 0,
            broken: false,
            started: false,
            start_on_kick: false,
            disabled: false,
            config_vector: VIRTIO_NO_VECTOR,
            vqs: Vec::new(),
            mem,
            transport: None,
        }
    }

    /// `virtio_init()`: names the device and sizes its config space.
    pub fn init(&mut self, name: &str, device_id: u16, config_size: usize) {
        self.name = name.to_owned();
        self.device_id = device_id;
        self.config = vec![0; config_size];
    }

    /// `virtio_add_queue()`: adds a queue offering `size` entries and returns its index.
    ///
    /// QEMU aborts when the queue count or size is out of range, this returns an error.
    pub fn add_queue(&mut self, size: u16) -> Result<u16> {
        if self.vqs.len() >= VIRTIO_QUEUE_MAX || size > VIRTQUEUE_MAX_SIZE {
            return Err(Error::generic(format!(
                "{}: cannot add a queue of size {size}",
                self.name
            )));
        }
        self.vqs.push(VirtQueue::new(size));
        Ok((self.vqs.len() - 1) as u16)
    }

    /// The QOM type name the device was initialised with.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The virtio device ID, 0 until [`init`](Self::init).
    pub fn device_id(&self) -> u16 {
        self.device_id
    }

    /// The raw config space as last filled in by the device model.
    pub fn config(&self) -> &[u8] {
        &self.config
    }

    /// `config_len`.
    pub fn config_len(&self) -> usize {
        self.config.len()
    }

    /// The features the device offers.
    pub fn host_features(&self) -> u64 {
        self.host_features
    }

    /// Replaces the offered features. Only meaningful before the device is plugged into a
    /// transport, like changing a feature property.
    pub fn set_host_features(&mut self, features: u64) {
        self.host_features = features;
    }

    /// Sets or clears one offered feature bit, like a `DEFINE_PROP_BIT64` feature property.
    pub fn set_host_feature(&mut self, bit: u32, on: bool) {
        if on {
            self.host_features |= feature(bit);
        } else {
            self.host_features &= !feature(bit);
        }
    }

    /// `virtio_host_has_feature()`.
    pub fn host_has_feature(&self, bit: u32) -> bool {
        has_feature(self.host_features, bit)
    }

    /// The features the driver accepted.
    pub fn guest_features(&self) -> u64 {
        self.guest_features
    }

    /// `virtio_vdev_has_feature()`: whether the driver accepted `bit`.
    pub fn has_feature(&self, bit: u32) -> bool {
        has_feature(self.guest_features, bit)
    }

    /// The device status byte.
    pub fn status(&self) -> u8 {
        self.status
    }

    /// Stores the status byte without any of the side effects of `virtio_set_status()`. Device
    /// models use it from [`VirtioDeviceClass::set_status`] when they need the new value in
    /// place before the core stores it, as virtio-rng does.
    pub fn set_status_value(&mut self, status: u8) {
        self.status = status;
    }

    /// The interrupt status byte.
    pub fn isr(&self) -> u8 {
        self.isr
    }

    /// `virtio_set_isr()`: ORs `value` into the interrupt status.
    pub fn set_isr(&mut self, value: u8) {
        self.isr |= value;
    }

    /// Clears the bits in `mask` from the interrupt status, what an interrupt acknowledge does.
    /// The transport should call [`update_irq`](Self::update_irq) afterwards.
    pub fn clear_isr(&mut self, mask: u8) {
        self.isr &= !mask;
    }

    /// The config space generation counter.
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// The selected queue.
    pub fn queue_sel(&self) -> u16 {
        self.queue_sel
    }

    /// Selects a queue.
    pub fn set_queue_sel(&mut self, sel: u16) {
        self.queue_sel = sel;
    }

    /// Whether [`error`](Self::error) was called since the last reset.
    pub fn is_broken(&self) -> bool {
        self.broken
    }

    /// Whether the driver has set `DRIVER_OK` (or kicked a legacy device before doing so).
    pub fn is_started(&self) -> bool {
        self.started
    }

    /// `virtio_device_disabled()`.
    pub fn is_disabled(&self) -> bool {
        self.disabled || self.broken
    }

    /// Disables or re-enables interrupts from the device, `vdev->disabled`.
    pub fn set_disabled(&mut self, disabled: bool) {
        self.disabled = disabled;
    }

    /// The config change vector.
    pub fn config_vector(&self) -> u16 {
        self.config_vector
    }

    /// Sets the config change vector.
    pub fn set_config_vector(&mut self, vector: u16) {
        self.config_vector = vector;
    }

    /// The guest memory the queues live in.
    pub fn mem(&self) -> &SharedGuestMemory {
        &self.mem
    }

    /// Connects the device to a transport, or disconnects it.
    pub fn set_transport(&mut self, transport: Option<Arc<dyn VirtioTransport>>) {
        self.transport = transport;
    }

    /// How many queues the device model added.
    pub fn num_queues(&self) -> usize {
        self.vqs.len()
    }

    /// Queue `n`, if the device has it.
    pub fn queue(&self, n: u16) -> Option<&VirtQueue> {
        self.vqs.get(usize::from(n))
    }

    /// `virtio_queue_get_num()`: 0 for queues the device does not have.
    pub fn queue_num(&self, n: u16) -> u16 {
        self.queue(n).map_or(0, |q| q.num)
    }

    /// `virtio_queue_get_max_num()`: the size offered for queue `n`, 0 if there is no such queue.
    pub fn queue_num_max(&self, n: u16) -> u16 {
        self.queue(n).map_or(0, |q| q.num_default)
    }

    /// `virtio_queue_get_addr()`: the descriptor table address of queue `n`.
    pub fn queue_addr(&self, n: u16) -> u64 {
        self.queue(n).map_or(0, |q| q.desc)
    }

    /// `virtio_queue_ready()`: whether the driver has told the device where the ring is.
    pub fn queue_ready(&self, n: u16) -> bool {
        self.queue(n).is_some_and(|q| q.avail != 0)
    }

    /// `virtio_queue_set_num()`.
    ///
    /// The guest cannot switch a queue between existing and not existing, cannot go above
    /// [`VIRTQUEUE_MAX_SIZE`], and going above what the device offers is a device error.
    pub fn set_queue_num(&mut self, n: u16, num: u32) {
        let Some(q) = self.vqs.get(usize::from(n)) else {
            return;
        };
        if (num != 0) != (q.num != 0) || num > u32::from(VIRTQUEUE_MAX_SIZE) {
            return;
        }
        if num > u32::from(q.num_default) {
            let max = q.num_default;
            self.error(&format!("virtio: queue {n} size {num} exceeds max size {max}"));
            return;
        }
        self.vqs[usize::from(n)].num = num as u16;
    }

    /// `virtio_queue_set_align()`: legacy only. Changes the alignment and recomputes the ring
    /// layout.
    pub fn set_queue_align(&mut self, n: u16, align: u32) {
        if self.has_feature(VIRTIO_F_VERSION_1) {
            error_report("tried to modify queue alignment for virtio-1 device");
            return;
        }
        let features = self.guest_features;
        if let Some(q) = self.vqs.get_mut(usize::from(n)) {
            if align != 0 {
                q.align = align;
                q.update_rings(features);
            }
        }
    }

    /// `virtio_queue_set_addr()`: legacy only. Places the ring at `addr`, or tears it down when
    /// `addr` is 0.
    pub fn set_queue_addr(&mut self, n: u16, addr: u64) {
        let features = self.guest_features;
        if let Some(q) = self.vqs.get_mut(usize::from(n)) {
            if q.num == 0 {
                return;
            }
            q.desc = addr;
            q.update_rings(features);
            if addr == 0 {
                q.rebuild(features);
            }
        }
    }

    /// `virtio_queue_update_rings()`.
    pub fn update_queue_rings(&mut self, n: u16) {
        let features = self.guest_features;
        if let Some(q) = self.vqs.get_mut(usize::from(n)) {
            q.update_rings(features);
        }
    }

    /// `virtio_queue_set_rings()`: the modern way to tell the device where the three parts of
    /// the ring are.
    pub fn set_queue_rings(&mut self, n: u16, desc: u64, avail: u64, used: u64) {
        let features = self.guest_features;
        if let Some(q) = self.vqs.get_mut(usize::from(n)) {
            if q.num == 0 {
                return;
            }
            q.desc = desc;
            q.avail = avail;
            q.used = used;
            q.rebuild(features);
        }
    }

    /// Sets the MSI vector of queue `n`.
    pub fn set_queue_vector(&mut self, n: u16, vector: u16) {
        if let Some(q) = self.vqs.get_mut(usize::from(n)) {
            q.vector = vector;
        }
    }

    /// `virtio_queue_reset()` without the device model hook: returns queue `n` to its state
    /// after a device reset.
    pub fn reset_queue(&mut self, n: u16) {
        if let Some(q) = self.vqs.get_mut(usize::from(n)) {
            q.reset();
        }
    }

    /// `virtio_queue_empty()`: whether queue `n` has nothing for the device.
    ///
    /// A broken or disabled device, a queue that is not set up and a ring that cannot be read
    /// all count as empty, so loops that drain a queue always end.
    pub fn queue_empty(&self, n: u16) -> bool {
        if self.is_disabled() {
            return true;
        }
        match self.queue(n).and_then(|q| q.ring.as_ref()) {
            Some(ring) => !ring.has_available(&*self.mem).unwrap_or(false),
            None => true,
        }
    }

    /// `virtqueue_pop()`: takes the next chain the driver made available on queue `n`.
    ///
    /// Anything wrong with the ring or the chain is a device error: the device is marked broken
    /// and `None` comes back.
    pub fn pop(&mut self, n: u16) -> Option<DescriptorChain> {
        if self.broken {
            return None;
        }
        let mem = Arc::clone(&self.mem);
        let q = self.vqs.get_mut(usize::from(n))?;
        if q.desc == 0 {
            return None;
        }
        let outcome = match q.ring.as_mut() {
            None => Err(match &q.ring_error {
                Some(e) => format!("virtio: invalid ring for queue {n}: {e}"),
                None => format!("virtio: queue {n} is not set up"),
            }),
            Some(ring) => match ring.has_available(&*mem) {
                Err(e) => Err(format!("virtio: {e}")),
                Ok(false) => Ok(None),
                Ok(true) if q.inuse >= u32::from(q.num) => {
                    Err("Virtqueue size exceeded".to_owned())
                }
                Ok(true) => match ring.pop(&*mem) {
                    Ok(Some(chain)) => {
                        q.inuse += 1;
                        Ok(Some(chain))
                    }
                    Ok(None) => Ok(None),
                    Err(e) => Err(format!("virtio: {e}")),
                },
            },
        };
        match outcome {
            Ok(chain) => chain,
            Err(msg) => {
                self.error(&msg);
                None
            }
        }
    }

    /// `virtqueue_push()`: returns `chain` to the driver on queue `n`, reporting `len` bytes
    /// written into it.
    pub fn push(&mut self, n: u16, chain: &DescriptorChain, len: u32) {
        let mem = Arc::clone(&self.mem);
        let Some(q) = self.vqs.get_mut(usize::from(n)) else {
            return;
        };
        let Some(ring) = q.ring.as_mut() else {
            return;
        };
        q.inuse = q.inuse.saturating_sub(1);
        if let Err(e) = ring.add_used(&*mem, chain, len) {
            self.error(&format!("virtio: {e}"));
        }
    }

    /// `virtqueue_detach_element()`: forgets a chain without returning it to the driver.
    pub fn detach(&mut self, n: u16, _chain: &DescriptorChain) {
        if let Some(q) = self.vqs.get_mut(usize::from(n)) {
            q.inuse = q.inuse.saturating_sub(1);
        }
    }

    /// `virtio_queue_get_notification()`.
    pub fn queue_notification(&self, n: u16) -> bool {
        self.queue(n).is_some_and(|q| q.notification)
    }

    /// `virtio_queue_set_notification()`: asks the driver to stop or resume kicking queue `n`.
    pub fn set_queue_notification(&mut self, n: u16, enable: bool) {
        let Some(q) = self.vqs.get_mut(usize::from(n)) else {
            return;
        };
        q.notification = enable;
        if let Some(ring) = q.ring.as_mut() {
            // Only a hint. A failure here shows up again on the next pop.
            let _ = ring.set_notification(&*self.mem, enable);
        }
    }

    /// `virtqueue_get_avail_bytes()`: how many device writable and device readable bytes the
    /// chains on queue `n` hold, looking no further than needed to reach `max_in` and `max_out`.
    /// Returns `(in_bytes, out_bytes)`, each capped at its maximum. Nothing in guest memory or in
    /// the queue changes.
    pub fn avail_bytes(&self, n: u16, max_in: u64, max_out: u64) -> (u64, u64) {
        if self.broken {
            return (0, 0);
        }
        let Some(ring) = self.queue(n).and_then(|q| q.ring.as_ref()) else {
            return (0, 0);
        };
        let mut peek = ring.clone();
        // With event index off, popping writes nothing to guest memory.
        peek.set_flags(false, has_feature(self.guest_features, VIRTIO_F_INDIRECT_DESC));
        let (mut in_total, mut out_total) = (0u64, 0u64);
        while let Ok(Some(chain)) = peek.pop(&*self.mem) {
            in_total += chain.writable_len();
            out_total += chain.readable_len();
            if in_total >= max_in && out_total >= max_out {
                break;
            }
        }
        (in_total.min(max_in), out_total.min(max_out))
    }

    /// `virtio_notify()`: raises a used buffer interrupt for queue `n` if the driver wants one.
    pub fn notify(&mut self, n: u16) {
        if self.should_notify(n) {
            self.set_isr(1);
            let vector = self.queue(n).map_or(VIRTIO_NO_VECTOR, |q| q.vector);
            self.notify_vector(vector);
        }
    }

    fn should_notify(&mut self, n: u16) -> bool {
        let notify_on_empty = self.has_feature(VIRTIO_F_NOTIFY_ON_EMPTY);
        let Some(q) = self.vqs.get_mut(usize::from(n)) else {
            return false;
        };
        let inuse = q.inuse;
        let Some(ring) = q.ring.as_mut() else {
            return false;
        };
        if notify_on_empty && inuse == 0 && !ring.has_available(&*self.mem).unwrap_or(false) {
            return true;
        }
        ring.needs_notification(&*self.mem).unwrap_or(false)
    }

    /// `virtio_notify_config()`: tells the driver the config space changed.
    pub fn notify_config(&mut self) {
        if self.status & VIRTIO_CONFIG_S_DRIVER_OK == 0 {
            return;
        }
        self.set_isr(3);
        self.generation = self.generation.wrapping_add(1);
        self.notify_vector(self.config_vector);
    }

    /// `virtio_update_irq()`: tells the transport to recompute its interrupt line.
    pub fn update_irq(&mut self) {
        self.notify_vector(VIRTIO_NO_VECTOR);
    }

    fn notify_vector(&self, vector: u16) {
        if self.disabled {
            return;
        }
        if let Some(t) = &self.transport {
            t.notify(vector, self.isr);
        }
    }

    /// `virtio_error()`: something the driver did makes the device unusable until reset.
    ///
    /// The message is reported, a virtio 1.0 driver sees `NEEDS_RESET` with a config interrupt,
    /// and the device stops processing queues.
    pub fn error(&mut self, msg: &str) {
        error_report(&format!("{}: {msg}", self.name));
        if self.has_feature(VIRTIO_F_VERSION_1) {
            self.status |= VIRTIO_CONFIG_S_NEEDS_RESET;
            self.notify_config();
        }
        self.broken = true;
    }

    fn rebuild_rings(&mut self) {
        let features = self.guest_features;
        for q in &mut self.vqs {
            q.rebuild(features);
        }
    }
}

/// A virtio device model, the virtual methods of `VirtioDeviceClass`.
///
/// Every method gets the core state as `vdev`. Only [`realize`](Self::realize) and
/// [`handle_output`](Self::handle_output) have to be written, the rest default to doing nothing
/// like a NULL class hook.
pub trait VirtioDeviceClass: Any + Send + fmt::Debug {
    /// Checks the properties and sets the device up: calls [`VirtIODevice::init`] and
    /// [`VirtIODevice::add_queue`].
    fn realize(&mut self, vdev: &mut VirtIODevice) -> Result<()>;

    /// Adds the device specific feature bits to what the transport and core offer.
    fn get_features(&mut self, _vdev: &VirtIODevice, features: u64) -> Result<u64> {
        Ok(features)
    }

    /// Last chance to refuse the features a virtio 1.0 driver picked, when it sets
    /// `FEATURES_OK`.
    fn validate_features(&mut self, _vdev: &VirtIODevice) -> Result<()> {
        Ok(())
    }

    /// The driver accepted `features`.
    fn set_features(&mut self, _vdev: &mut VirtIODevice, _features: u64) {}

    /// Fills in the config space before the driver reads it.
    fn get_config(&mut self, _vdev: &VirtIODevice, _config: &mut [u8]) {}

    /// The driver wrote the config space. The device may change `config`.
    fn set_config(&mut self, _vdev: &mut VirtIODevice, _config: &mut [u8]) {}

    /// The driver is changing the status to `status`. The core stores it after this returns.
    fn set_status(&mut self, _vdev: &mut VirtIODevice, _status: u8) -> Result<()> {
        Ok(())
    }

    /// Device specific part of a device reset.
    fn reset(&mut self, _vdev: &mut VirtIODevice) {}

    /// The driver kicked `queue`.
    fn handle_output(&mut self, vdev: &mut VirtIODevice, queue: u16);

    /// For downcasting to the concrete model.
    fn as_any(&self) -> &dyn Any;

    /// For downcasting to the concrete model.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// A realized device: the core state and the model that goes with it.
///
/// This is what a transport holds. Everything a transport register write can trigger goes
/// through here.
#[derive(Debug)]
pub struct VirtioBackend {
    vdev: VirtIODevice,
    class: Box<dyn VirtioDeviceClass>,
}

impl VirtioBackend {
    /// Realizes `class` on guest memory `mem`.
    pub fn new(mut class: Box<dyn VirtioDeviceClass>, mem: SharedGuestMemory) -> Result<Self> {
        let mut vdev = VirtIODevice::new(mem);
        class.realize(&mut vdev)?;
        Ok(VirtioBackend { vdev, class })
    }

    /// The core state.
    pub fn vdev(&self) -> &VirtIODevice {
        &self.vdev
    }

    /// The core state, mutably.
    pub fn vdev_mut(&mut self) -> &mut VirtIODevice {
        &mut self.vdev
    }

    /// The device model.
    pub fn class(&self) -> &dyn VirtioDeviceClass {
        &*self.class
    }

    /// The core state and the device model together.
    pub fn parts_mut(&mut self) -> (&mut VirtIODevice, &mut dyn VirtioDeviceClass) {
        (&mut self.vdev, &mut *self.class)
    }

    /// The core state and the device model as its concrete type, if it is a `D`.
    pub fn downcast_mut<D: VirtioDeviceClass>(&mut self) -> Option<(&mut VirtIODevice, &mut D)> {
        let dev = self.class.as_any_mut().downcast_mut::<D>()?;
        Some((&mut self.vdev, dev))
    }

    /// `virtio_bus_device_plugged()`: connects the device to `transport`, which adds
    /// `transport_features` first (the transport's `pre_plugged` hook), and then asks the model
    /// for its features.
    pub fn plug(
        &mut self,
        transport: Arc<dyn VirtioTransport>,
        transport_features: u64,
    ) -> Result<()> {
        self.vdev.host_features |= transport_features;
        let features = self.class.get_features(&self.vdev, self.vdev.host_features)?;
        self.vdev.host_features = features;
        self.vdev.transport = Some(transport);
        Ok(())
    }

    fn validate_features(&mut self) -> Result<()> {
        if self.vdev.host_has_feature(VIRTIO_F_IOMMU_PLATFORM)
            && !self.vdev.has_feature(VIRTIO_F_IOMMU_PLATFORM)
        {
            return Err(Error::generic("the driver did not accept VIRTIO_F_IOMMU_PLATFORM"));
        }
        self.class.validate_features(&self.vdev)
    }

    /// `virtio_set_status()`.
    ///
    /// When a virtio 1.0 driver sets `FEATURES_OK` the features are validated first, and if
    /// that fails the status is left alone so the driver sees `FEATURES_OK` did not stick.
    pub fn set_status(&mut self, val: u8) -> Result<()> {
        if self.vdev.has_feature(VIRTIO_F_VERSION_1)
            && self.vdev.status & VIRTIO_CONFIG_S_FEATURES_OK == 0
            && val & VIRTIO_CONFIG_S_FEATURES_OK != 0
        {
            self.validate_features()?;
        }
        if (self.vdev.status ^ val) & VIRTIO_CONFIG_S_DRIVER_OK != 0 {
            self.set_started(val & VIRTIO_CONFIG_S_DRIVER_OK != 0);
        }
        if let Err(e) = self.class.set_status(&mut self.vdev, val) {
            warn_report(&format!(
                "virtio: {} failed to change status to {val:#x}: {e}",
                self.vdev.name
            ));
        }
        self.vdev.status = val;
        Ok(())
    }

    fn set_started(&mut self, started: bool) {
        if started {
            self.vdev.start_on_kick = false;
        }
        self.vdev.started = started;
    }

    /// `virtio_set_features_nocheck()`: stores what the driver accepted, dropping bits the
    /// device never offered. Offering extra bits is an error, but the rest still take effect.
    fn set_features_nocheck(&mut self, val: u64) -> Result<()> {
        let extra = val & !self.vdev.host_features;
        let val = val & self.vdev.host_features;
        self.class.set_features(&mut self.vdev, val);
        self.vdev.guest_features = val;
        self.vdev.rebuild_rings();
        if extra != 0 {
            Err(Error::generic(format!(
                "virtio: {} got unsupported features {extra:#x}",
                self.vdev.name
            )))
        } else {
            Ok(())
        }
    }

    /// `virtio_set_features()`: what the driver writes as its accepted features. Refused once
    /// `FEATURES_OK` is set.
    pub fn set_features(&mut self, val: u64) -> Result<()> {
        if self.vdev.status & VIRTIO_CONFIG_S_FEATURES_OK != 0 {
            return Err(Error::generic(format!(
                "virtio: {} features cannot change after FEATURES_OK",
                self.vdev.name
            )));
        }
        let ret = self.set_features_nocheck(val);
        if ret.is_ok()
            && !(self.vdev.started || self.vdev.status & VIRTIO_CONFIG_S_DRIVER_OK != 0)
            && !self.vdev.has_feature(VIRTIO_F_VERSION_1)
        {
            self.vdev.start_on_kick = true;
        }
        ret
    }

    /// `virtio_reset()`: the device goes back to how it was when plugged in.
    pub fn reset(&mut self) {
        let _ = self.set_status(0);
        self.class.reset(&mut self.vdev);
        let vdev = &mut self.vdev;
        vdev.start_on_kick = false;
        vdev.started = false;
        vdev.broken = false;
        let _ = self.set_features_nocheck(0);
        let vdev = &mut self.vdev;
        vdev.queue_sel = 0;
        vdev.status = 0;
        vdev.disabled = false;
        vdev.isr = 0;
        vdev.config_vector = VIRTIO_NO_VECTOR;
        vdev.notify_vector(vdev.config_vector);
        for q in &mut vdev.vqs {
            q.reset();
        }
    }

    /// `virtio_config_read{b,w,l}()` and the modern variants: reads `size` bytes (1, 2 or 4) of
    /// config space at `addr`, little endian. Reads past the end return all ones.
    pub fn config_read(&mut self, addr: u64, size: u32) -> u32 {
        let len = self.config_len_u64();
        if addr.checked_add(u64::from(size)).is_none_or(|end| end > len) {
            return u32::MAX;
        }
        let mut config = std::mem::take(&mut self.vdev.config);
        self.class.get_config(&self.vdev, &mut config);
        self.vdev.config = config;
        let start = addr as usize;
        let mut b = [0u8; 4];
        b[..size as usize].copy_from_slice(&self.vdev.config[start..start + size as usize]);
        u32::from_le_bytes(b)
    }

    /// `virtio_config_write{b,w,l}()` and the modern variants: writes the low `size` bytes of
    /// `val` to config space at `addr`. Writes past the end are dropped.
    pub fn config_write(&mut self, addr: u64, size: u32, val: u32) {
        let len = self.config_len_u64();
        if addr.checked_add(u64::from(size)).is_none_or(|end| end > len) {
            return;
        }
        let start = addr as usize;
        self.vdev.config[start..start + size as usize]
            .copy_from_slice(&val.to_le_bytes()[..size as usize]);
        let mut config = std::mem::take(&mut self.vdev.config);
        self.class.set_config(&mut self.vdev, &mut config);
        self.vdev.config = config;
    }

    fn config_len_u64(&self) -> u64 {
        self.vdev.config.len() as u64
    }

    /// `virtio_queue_notify()`: the driver kicked queue `n`.
    pub fn queue_notify(&mut self, n: u16) {
        let Some(q) = self.vdev.queue(n) else {
            return;
        };
        if q.desc == 0 || self.vdev.broken {
            return;
        }
        self.class.handle_output(&mut self.vdev, n);
        if self.vdev.start_on_kick {
            self.set_started(true);
        }
    }
}
