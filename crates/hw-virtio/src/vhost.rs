// SPDX-License-Identifier: GPL-2.0-or-later

//! The device side of vhost, a port of the parts of `hw/virtio/vhost.c` and the backend init of
//! `hw/virtio/vhost-user.c` that vhost devices share.
//!
//! [`VhostDev`] is QEMU's `struct vhost_dev`: it owns the connection to a backend (the kernel
//! through `ruvm_vhost::kernel::VhostKernel`, or another process through
//! `ruvm_vhost::user::Frontend`), negotiates features, and starts and stops the rings. The
//! message sequences follow QEMU:
//!
//! - init: for vhost-user, `GET_FEATURES`, then if the backend has protocol features
//!   `GET_PROTOCOL_FEATURES`, `SET_PROTOCOL_FEATURES` and, with `MQ`, `GET_QUEUE_NUM`. Then for
//!   both kinds `SET_OWNER`, `GET_FEATURES`, and `SET_VRING_CALL` and `SET_VRING_ERR` for every
//!   ring.
//! - start: `SET_FEATURES` with what the driver accepted, `SET_MEM_TABLE`, then for every ring
//!   the driver set up `SET_VRING_NUM`, `SET_VRING_BASE`, `SET_VRING_ADDR`, `SET_VRING_KICK` and
//!   `SET_VRING_CALL`, and last `SET_VRING_ENABLE` on every ring when the backend has protocol
//!   features.
//! - stop: `SET_VRING_ENABLE` off on every ring (same condition), then `GET_VRING_BASE` for
//!   every started ring, whose answer goes back into the device's queue state along with the
//!   used index the backend left in guest memory.
//!
//! Differences from QEMU:
//!
//! - There is no memory listener. Whoever builds the machine hands the device its memory table
//!   with [`VhostDev::set_mem_table`], and ring addresses are translated through that table.
//! - Kicks and interrupts go through [`Notifier`]s that are driven by hand: the device forwards
//!   a guest kick with [`VhostDev::kick`] and [`VhostDev::poll_calls`] turns backend signals
//!   into interrupts. There is no ioeventfd or irqfd, and one call notifier serves as both the
//!   masked and the unmasked one. On Linux the notifiers are eventfds. Elsewhere they are
//!   socket pairs, which carry the same eight byte writes.
//! - vhost-user negotiates `MQ`, `REPLY_ACK`, `RESET_DEVICE` and, for devices that read config
//!   space from the backend, `CONFIG`. QEMU takes every protocol feature it knows. Dirty
//!   logging, the IOTLB, inflight tracking, `STATUS`, the backend request channel and
//!   migration are not ported.
//! - `SET_FEATURES` carries only bits the backend offered. QEMU sends every bit the driver
//!   accepted.

use std::fmt;
use std::fs::File;
use std::io::{self, Read, Write};
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::Arc;

use ruvm_base::error::strerror;
use ruvm_base::{Error, Result};
#[cfg(target_os = "linux")]
use ruvm_vhost::kernel::VhostKernel;
use ruvm_vhost::user::Frontend;
use ruvm_vhost::user::message::protocol;
use ruvm_vhost::{MemoryRegion, VHOST_USER_F_PROTOCOL_FEATURES, VhostBackend, VringAddr};
use ruvm_virtio_queue::VIRTIO_F_RING_PACKED;

use crate::virtio::{VIRTIO_F_IOMMU_PLATFORM, VirtIODevice, feature, has_feature};

/// The `chardev` of a vhost-user device: where the backend listens.
#[derive(Debug)]
pub enum VhostUserChardev {
    /// Connect to the Unix socket at this path, like `-chardev socket,path=...`.
    Path(PathBuf),
    /// A socket that is already connected.
    Stream(UnixStream),
}

impl VhostUserChardev {
    /// Connects, `vhost_user_init()` plus the chardev's own connect.
    pub fn connect(self) -> Result<Frontend> {
        match self {
            VhostUserChardev::Path(path) => match UnixStream::connect(&path) {
                Ok(stream) => Ok(Frontend::new(stream)),
                Err(e) => Err(Error::generic(format!(
                    "Failed to connect to '{}': {}",
                    path.display(),
                    strerror(&e)
                ))),
            },
            VhostUserChardev::Stream(stream) => Ok(Frontend::new(stream)),
        }
    }
}

/// One region of guest memory for [`VhostDev::set_mem_table`].
///
/// Kernel backends read `userspace_addr` directly. vhost-user backends map `fd` from
/// `mmap_offset`, so every region needs a file for them.
#[derive(Clone, Debug)]
pub struct VhostMemRegion {
    /// Guest physical address of the start of the region.
    pub guest_phys_addr: u64,
    /// Size in bytes.
    pub memory_size: u64,
    /// Where the region is mapped in this process.
    pub userspace_addr: u64,
    /// Offset of the region in `fd`.
    pub mmap_offset: u64,
    /// The file backing the region.
    pub fd: Option<Arc<OwnedFd>>,
}

impl VhostMemRegion {
    fn translate(&self, gpa: u64, len: u64) -> Option<u64> {
        let off = gpa.checked_sub(self.guest_phys_addr)?;
        if off.checked_add(len)? > self.memory_size {
            return None;
        }
        self.userspace_addr.checked_add(off)
    }
}

/// An event notifier, QEMU's `EventNotifier`: the file a backend kicks or signals, and our end
/// of it.
pub struct Notifier {
    local: File,
    remote: Option<OwnedFd>,
}

impl fmt::Debug for Notifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Notifier").field("fd", &self.fd()).finish()
    }
}

impl Notifier {
    /// A new notifier with nothing pending.
    #[cfg(target_os = "linux")]
    pub fn new() -> io::Result<Self> {
        use rustix::event::{EventfdFlags, eventfd};
        let fd = eventfd(0, EventfdFlags::CLOEXEC | EventfdFlags::NONBLOCK)?;
        Ok(Notifier { local: File::from(fd), remote: None })
    }

    /// A new notifier with nothing pending.
    #[cfg(not(target_os = "linux"))]
    pub fn new() -> io::Result<Self> {
        let (local, remote) = UnixStream::pair()?;
        local.set_nonblocking(true)?;
        Ok(Notifier {
            local: File::from(OwnedFd::from(local)),
            remote: Some(OwnedFd::from(remote)),
        })
    }

    /// The file handed to the backend.
    pub fn fd(&self) -> BorrowedFd<'_> {
        match &self.remote {
            Some(fd) => fd.as_fd(),
            None => self.local.as_fd(),
        }
    }

    /// `event_notifier_set()`: signals the backend.
    pub fn notify(&self) -> io::Result<()> {
        (&self.local).write_all(&1u64.to_ne_bytes())
    }

    /// `event_notifier_test_and_clear()`: whether the backend signalled since the last call.
    pub fn test_and_clear(&self) -> bool {
        let mut signalled = false;
        let mut buf = [0u8; 8];
        loop {
            match (&self.local).read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    signalled = true;
                    if self.remote.is_none() {
                        // An eventfd read takes the whole count at once.
                        break;
                    }
                }
            }
        }
        signalled
    }
}

/// The connection to a vhost backend.
#[derive(Debug)]
pub enum VhostConnection {
    /// An in-kernel backend such as `/dev/vhost-vsock`.
    #[cfg(target_os = "linux")]
    Kernel(VhostKernel),
    /// A vhost-user backend in another process.
    User(Frontend),
}

impl VhostConnection {
    fn ops(&mut self) -> &mut dyn VhostBackend {
        match self {
            #[cfg(target_os = "linux")]
            VhostConnection::Kernel(k) => k,
            VhostConnection::User(u) => u,
        }
    }

    fn user(&mut self) -> Option<&mut Frontend> {
        match self {
            #[cfg(target_os = "linux")]
            VhostConnection::Kernel(_) => None,
            VhostConnection::User(u) => Some(u),
        }
    }

    fn is_user(&self) -> bool {
        matches!(self, VhostConnection::User(_))
    }
}

/// Options for [`VhostDev::new`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VhostDevOptions {
    /// The device reads its config space from a vhost-user backend, so the backend must offer
    /// `VHOST_USER_PROTOCOL_F_CONFIG`.
    pub supports_config: bool,
}

/// `struct vhost_virtqueue`: the notifiers of one ring.
#[derive(Debug)]
struct VhostVirtqueue {
    kick: Notifier,
    call: Notifier,
    err: Notifier,
    started: bool,
}

/// `struct vhost_dev`: a vhost backend driving some of a device's queues.
#[derive(Debug)]
pub struct VhostDev {
    conn: VhostConnection,
    vq_index: u32,
    features: u64,
    acked_features: u64,
    protocol_features: u64,
    max_queues: u64,
    mem: Vec<VhostMemRegion>,
    vqs: Vec<VhostVirtqueue>,
    started: bool,
}

fn vhost_err(what: &str, e: impl fmt::Display) -> Error {
    Error::generic(format!("{what}: {e}"))
}

fn io_err(what: &str, e: io::Error) -> Error {
    Error::generic(format!("{what}: {e}"))
}

impl VhostDev {
    /// `vhost_dev_init()`: connects `nvqs` rings, the device's queues `0..nvqs`, to the
    /// backend behind `conn`.
    pub fn new(conn: VhostConnection, nvqs: usize, options: VhostDevOptions) -> Result<Self> {
        let mut dev = VhostDev {
            conn,
            vq_index: 0,
            features: 0,
            acked_features: 0,
            protocol_features: 0,
            max_queues: 1,
            mem: Vec::new(),
            vqs: Vec::with_capacity(nvqs),
            started: false,
        };
        if let Some(u) = dev.conn.user() {
            let (protocol_features, max_queues) = Self::user_backend_init(u, options)?;
            dev.protocol_features = protocol_features;
            dev.max_queues = max_queues;
        }
        dev.conn.ops().set_owner().map_err(|e| vhost_err("vhost_set_owner failed", e))?;
        dev.features = dev
            .conn
            .ops()
            .get_features()
            .map_err(|e| vhost_err("vhost_init_features failed", e))?;
        for i in 0..nvqs {
            let vq = dev
                .virtqueue_init(i as u32)
                .map_err(|e| vhost_err(&format!("Failed to initialize virtqueue {i}"), e))?;
            dev.vqs.push(vq);
        }
        Ok(dev)
    }

    /// `vhost_user_backend_init()`: protocol feature negotiation. Returns the agreed protocol
    /// features and the number of queues the backend supports.
    fn user_backend_init(u: &mut Frontend, options: VhostDevOptions) -> Result<(u64, u64)> {
        let features = u.get_features().map_err(|e| vhost_err("vhost_backend_init failed", e))?;
        if features & VHOST_USER_F_PROTOCOL_FEATURES == 0 {
            return Ok((0, 1));
        }
        let offered =
            u.get_protocol_features().map_err(|e| vhost_err("vhost_backend_init failed", e))?;
        let mut supported = protocol::MQ | protocol::REPLY_ACK | protocol::RESET_DEVICE;
        if options.supports_config {
            if offered & protocol::CONFIG == 0 {
                return Err(Error::generic(
                    "vhost-user device expecting VHOST_USER_PROTOCOL_F_CONFIG but the vhost-user \
                     backend does not support it.",
                ));
            }
            supported |= protocol::CONFIG;
        }
        let agreed = offered & supported;
        u.set_protocol_features(agreed).map_err(|e| vhost_err("vhost_backend_init failed", e))?;
        let max_queues = if agreed & protocol::MQ != 0 {
            u.get_queue_num().map_err(|e| vhost_err("vhost_backend_init failed", e))?
        } else {
            1
        };
        Ok((agreed, max_queues))
    }

    /// `vhost_virtqueue_init()`.
    fn virtqueue_init(&mut self, n: u32) -> io::Result<VhostVirtqueue> {
        let vq = VhostVirtqueue {
            kick: Notifier::new()?,
            call: Notifier::new()?,
            err: Notifier::new()?,
            started: false,
        };
        let index = self.vq_index + n;
        let ops = self.conn.ops();
        ops.set_vring_call(index, Some(vq.call.fd())).map_err(to_io)?;
        ops.set_vring_err(index, Some(vq.err.fd())).map_err(to_io)?;
        Ok(vq)
    }

    /// The connection, for backend specific requests like the vsock ioctls.
    pub fn connection(&mut self) -> &mut VhostConnection {
        &mut self.conn
    }

    /// The features the backend offered at init.
    pub fn features(&self) -> u64 {
        self.features
    }

    /// The features last sent with `SET_FEATURES`.
    pub fn acked_features(&self) -> u64 {
        self.acked_features
    }

    /// The vhost-user protocol features agreed at init, zero for kernel backends.
    pub fn protocol_features(&self) -> u64 {
        self.protocol_features
    }

    /// The number of queues a vhost-user backend with `MQ` supports, 1 otherwise.
    pub fn max_queues(&self) -> u64 {
        self.max_queues
    }

    /// The number of rings the backend runs.
    pub fn nvqs(&self) -> usize {
        self.vqs.len()
    }

    /// Whether the rings are running, `vhost_dev_is_started()`.
    pub fn is_started(&self) -> bool {
        self.started
    }

    /// `vhost_get_features()`: clears every bit of `feature_bits` from `features` that the
    /// backend did not offer.
    pub fn get_features(&self, feature_bits: &[u32], features: u64) -> u64 {
        let mut features = features;
        for &bit in feature_bits {
            if !has_feature(self.features, bit) {
                features &= !feature(bit);
            }
        }
        features
    }

    /// Sets the guest memory the backend sees. Sent to the backend right away if it is
    /// running, and on every start.
    pub fn set_mem_table(&mut self, regions: Vec<VhostMemRegion>) -> Result<()> {
        self.mem = regions;
        if self.started {
            self.send_mem_table()?;
        }
        Ok(())
    }

    /// The memory table.
    pub fn mem_table(&self) -> &[VhostMemRegion] {
        &self.mem
    }

    fn send_mem_table(&mut self) -> Result<()> {
        let regions: Vec<MemoryRegion<'_>> = self
            .mem
            .iter()
            .map(|r| MemoryRegion {
                guest_phys_addr: r.guest_phys_addr,
                memory_size: r.memory_size,
                userspace_addr: r.userspace_addr,
                mmap_offset: r.mmap_offset,
                fd: r.fd.as_deref().map(AsFd::as_fd),
            })
            .collect();
        let conn = &mut self.conn;
        let ops = conn.ops();
        ops.set_mem_table(&regions).map_err(|e| vhost_err("vhost_set_mem_table failed", e))
    }

    fn translate(&self, gpa: u64, len: u64) -> Option<u64> {
        self.mem.iter().find_map(|r| r.translate(gpa, len))
    }

    /// `vhost_dev_start()`: hands the rings to the backend. `acked_features` are the features
    /// the driver accepted.
    pub fn start(&mut self, vdev: &mut VirtIODevice, acked_features: u64) -> Result<()> {
        self.started = true;
        self.acked_features = acked_features;
        let result = self.do_start(vdev);
        if result.is_err() {
            for i in 0..self.vqs.len() {
                if self.vqs[i].started {
                    let _ = self.virtqueue_stop(vdev, i);
                }
            }
            self.started = false;
        }
        result
    }

    fn do_start(&mut self, vdev: &mut VirtIODevice) -> Result<()> {
        // No IOMMU is modelled, so the backend never gets VIRTIO_F_IOMMU_PLATFORM. Transport
        // bits the backend never offered (VIRTIO_F_ANY_LAYOUT, say) are dropped too, where
        // QEMU would pass them on.
        let features = self.acked_features & self.features & !feature(VIRTIO_F_IOMMU_PLATFORM);
        self.conn.ops().set_features(features).map_err(|e| vhost_err("vhost_set_features", e))?;
        self.send_mem_table()?;
        for i in 0..self.vqs.len() {
            self.virtqueue_start(vdev, i)?;
        }
        self.set_vring_enable(true).map_err(|e| vhost_err("vhost_set_vring_enable", e))?;
        Ok(())
    }

    /// `vhost_dev_set_vring_enable()`: only vhost-user backends that negotiated protocol
    /// features have rings that start disabled.
    fn set_vring_enable(&mut self, enable: bool) -> ruvm_vhost::Result<()> {
        if self.features & VHOST_USER_F_PROTOCOL_FEATURES == 0 {
            return Ok(());
        }
        let base = self.vq_index;
        let nvqs = self.vqs.len() as u32;
        if let Some(u) = self.conn.user() {
            for i in 0..nvqs {
                u.set_vring_enable(base + i, enable)?;
            }
        }
        Ok(())
    }

    /// `vhost_virtqueue_start()`.
    fn virtqueue_start(&mut self, vdev: &mut VirtIODevice, i: usize) -> Result<()> {
        let n = self.vq_index + i as u32;
        let qn = n as u16;
        let Some(q) = vdev.queue(qn) else {
            return Ok(());
        };
        let (num, desc, avail, used) = (q.num(), q.desc(), q.avail(), q.used());
        if desc == 0 {
            // The driver did not set this ring up.
            return Ok(());
        }
        let size = u64::from(num);
        let (desc_len, avail_len, used_len) = if vdev.has_feature(VIRTIO_F_RING_PACKED) {
            (16 * size, 4, 4)
        } else {
            (16 * size, 6 + 2 * size, 6 + 8 * size)
        };
        let desc_uva = self
            .translate(desc, desc_len)
            .ok_or_else(|| Error::generic(format!("Unable to map desc ring for ring {i}")))?;
        let avail_uva = self
            .translate(avail, avail_len)
            .ok_or_else(|| Error::generic(format!("Unable to map avail ring for ring {i}")))?;
        let used_uva = self
            .translate(used, used_len)
            .ok_or_else(|| Error::generic(format!("Unable to map used ring for ring {i}")))?;
        let base = vdev.last_avail_idx(qn);
        let conn = &mut self.conn;
        let vq = &self.vqs[i];
        let ops = conn.ops();
        ops.set_vring_num(n, u32::from(num)).map_err(|e| vhost_err("vhost_set_vring_num", e))?;
        ops.set_vring_base(n, base).map_err(|e| vhost_err("vhost_set_vring_base", e))?;
        ops.set_vring_addr(&VringAddr {
            index: n,
            flags: 0,
            desc_user_addr: desc_uva,
            used_user_addr: used_uva,
            avail_user_addr: avail_uva,
            log_guest_addr: used,
        })
        .map_err(|e| vhost_err("vhost_set_vring_addr", e))?;
        ops.set_vring_kick(n, Some(vq.kick.fd()))
            .map_err(|e| vhost_err("vhost_set_vring_kick", e))?;
        // Throw away anything signalled before the ring started.
        vq.call.test_and_clear();
        ops.set_vring_call(n, Some(vq.call.fd()))
            .map_err(|e| vhost_err("vhost_set_vring_call", e))?;
        self.vqs[i].started = true;
        Ok(())
    }

    /// `vhost_dev_stop()`: takes the rings back from the backend.
    pub fn stop(&mut self, vdev: &mut VirtIODevice) -> Result<()> {
        let mut result =
            self.set_vring_enable(false).map_err(|e| vhost_err("vhost_set_vring_enable", e));
        for i in 0..self.vqs.len() {
            if let Err(e) = self.virtqueue_stop(vdev, i) {
                if result.is_ok() {
                    result = Err(e);
                }
            }
        }
        self.started = false;
        result
    }

    /// `vhost_virtqueue_stop()`.
    fn virtqueue_stop(&mut self, vdev: &mut VirtIODevice, i: usize) -> Result<()> {
        let n = self.vq_index + i as u32;
        let qn = n as u16;
        if vdev.queue(qn).is_none_or(|q| q.desc() == 0) {
            return Ok(());
        }
        self.vqs[i].started = false;
        match self.conn.ops().get_vring_base(n) {
            Ok(base) => {
                vdev.set_last_avail_idx(qn, base);
                vdev.update_used_idx(qn);
                Ok(())
            }
            Err(e) => {
                vdev.restore_last_avail_idx(qn);
                Err(vhost_err(&format!("vhost VQ {i} ring restore failed"), e))
            }
        }
    }

    /// Passes a guest kick of device queue `queue` on to the backend.
    pub fn kick(&self, queue: u16) -> Result<()> {
        let Some(vq) =
            u32::from(queue).checked_sub(self.vq_index).and_then(|i| self.vqs.get(i as usize))
        else {
            return Ok(());
        };
        vq.kick.notify().map_err(|e| io_err("vhost kick failed", e))
    }

    /// The notifier the backend kicks on ring `i`, for tests and event loops.
    pub fn kick_notifier(&self, i: usize) -> Option<&Notifier> {
        self.vqs.get(i).map(|vq| &vq.kick)
    }

    /// The notifier the backend signals when ring `i` has used buffers.
    pub fn call_notifier(&self, i: usize) -> Option<&Notifier> {
        self.vqs.get(i).map(|vq| &vq.call)
    }

    /// The notifier the backend signals when ring `i` hits an error.
    pub fn err_notifier(&self, i: usize) -> Option<&Notifier> {
        self.vqs.get(i).map(|vq| &vq.err)
    }

    /// Interrupts the driver for every started ring whose call notifier was signalled, what
    /// an irqfd would do. Returns whether any was.
    pub fn poll_calls(&self, vdev: &mut VirtIODevice) -> bool {
        let mut any = false;
        for (i, vq) in self.vqs.iter().enumerate() {
            if vq.started && vq.call.test_and_clear() {
                vdev.notify_irqfd((self.vq_index + i as u32) as u16);
                any = true;
            }
        }
        any
    }

    /// Whether this is a vhost-user backend.
    pub fn is_user(&self) -> bool {
        self.conn.is_user()
    }

    /// `vhost_dev_get_config()`: reads `len` bytes of config space from a vhost-user backend.
    pub fn get_config(&mut self, len: usize) -> Result<Vec<u8>> {
        match &mut self.conn {
            VhostConnection::User(u) => {
                u.get_config(0, len as u32, 0).map_err(|e| vhost_err("vhost_get_config failed", e))
            }
            #[cfg(target_os = "linux")]
            VhostConnection::Kernel(_) => Err(Error::generic("vhost_get_config not supported")),
        }
    }
}

fn to_io(e: ruvm_vhost::Error) -> io::Error {
    match e {
        ruvm_vhost::Error::Io(e) => e,
        e => io::Error::other(e.to_string()),
    }
}
