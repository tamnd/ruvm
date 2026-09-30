// SPDX-License-Identifier: GPL-2.0-or-later

//! vhost-vsock and vhost-user-vsock, ports of `hw/virtio/vhost-vsock-common.c`,
//! `hw/virtio/vhost-vsock.c` and `hw/virtio/vhost-user-vsock.c`.
//!
//! Both devices have three queues of 128 entries: receive and transmit, which the vhost
//! backend runs, and the event queue, which stays with the device and only ever carries the
//! transport reset event ([`VhostVsock::send_transport_reset`]). The config space is the
//! 64-bit guest CID.
//!
//! [`VhostVsock`] uses the kernel's `/dev/vhost-vsock`, which only exists on Linux. On other
//! hosts realize fails the way QEMU does when the file is missing. [`VhostUserVsock`] talks to
//! a vhost-user backend such as `vhost-device-vsock` and reads the config space from it.
//!
//! Differences from QEMU:
//!
//! - There is no ioeventfd or irqfd. A guest kick on a vhost queue reaches the device's
//!   `handle_output`, which passes it to the backend, and the owner of the device calls
//!   [`VhostVsock::poll_calls`] (or the vhost-user variant) when the backend signals. See the
//!   [`vhost`](crate::vhost) module.
//! - The guest memory table comes from the caller through `set_mem_table`.
//! - `vhostfd` takes an open file and does not switch it to non-blocking mode.
//! - Migration (the post load transport reset) and the vhost-user config change notifier are
//!   not ported.

use std::any::Any;
use std::fs::File;

#[cfg(target_os = "linux")]
use ruvm_base::error::strerror;
use ruvm_base::{Error, Result, error_report};
#[cfg(target_os = "linux")]
use ruvm_vhost::kernel::VhostKernel;
use ruvm_virtio_queue::VIRTIO_F_RING_PACKED;

use crate::vhost::{VhostConnection, VhostDev, VhostDevOptions, VhostMemRegion, VhostUserChardev};
use crate::virtio::{
    VIRTIO_CONFIG_S_DRIVER_OK, VIRTIO_F_IN_ORDER, VIRTIO_F_NOTIFICATION_DATA,
    VIRTIO_F_NOTIFY_ON_EMPTY, VIRTIO_F_RING_RESET, VIRTIO_F_VERSION_1, VirtIODevice,
    VirtioDeviceClass, feature, has_feature,
};

/// `TYPE_VHOST_VSOCK`.
pub const TYPE_VHOST_VSOCK: &str = "vhost-vsock-device";
/// `TYPE_VHOST_USER_VSOCK`.
pub const TYPE_VHOST_USER_VSOCK: &str = "vhost-user-vsock-device";
/// `VIRTIO_ID_VSOCK`.
pub const VIRTIO_ID_VSOCK: u16 = 19;
/// `VHOST_VSOCK_QUEUE_SIZE`.
pub const VHOST_VSOCK_QUEUE_SIZE: u16 = 128;
/// `VIRTIO_VSOCK_F_SEQPACKET`.
pub const VIRTIO_VSOCK_F_SEQPACKET: u32 = 0;
/// `VIRTIO_VSOCK_EVENT_TRANSPORT_RESET`.
pub const VIRTIO_VSOCK_EVENT_TRANSPORT_RESET: u32 = 0;
/// The size of `struct virtio_vsock_config`.
pub const VIRTIO_VSOCK_CONFIG_SIZE: usize = 8;
/// Where the kernel backend lives.
pub const VHOST_VSOCK_PATH: &str = "/dev/vhost-vsock";

const VIRTIO_RING_F_INDIRECT_DESC: u32 = 28;
const VIRTIO_RING_F_EVENT_IDX: u32 = 29;

/// The queue the device keeps for itself.
const EVENT_QUEUE: u16 = 2;
/// The queues the backend runs, receive and transmit.
const VHOST_NVQS: usize = 2;

/// `feature_bits` in `vhost-vsock-common.c`.
const FEATURE_BITS: &[u32] = &[VIRTIO_VSOCK_F_SEQPACKET, VIRTIO_F_RING_RESET, VIRTIO_F_RING_PACKED];

/// `user_feature_bits` in `vhost-user-vsock.c`.
const USER_FEATURE_BITS: &[u32] = &[
    VIRTIO_F_VERSION_1,
    VIRTIO_RING_F_INDIRECT_DESC,
    VIRTIO_RING_F_EVENT_IDX,
    VIRTIO_F_NOTIFY_ON_EMPTY,
    VIRTIO_F_IN_ORDER,
    VIRTIO_F_NOTIFICATION_DATA,
];

/// QEMU's `OnOffAuto` property type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OnOffAuto {
    /// Use the feature if the backend has it.
    #[default]
    Auto,
    /// Require the feature.
    On,
    /// Never offer the feature.
    Off,
}

/// `VHostVSockCommon`: what both vsock devices share.
#[derive(Debug, Default)]
struct VsockCommon {
    vhost: Option<VhostDev>,
    seqpacket: OnOffAuto,
    mem: Vec<VhostMemRegion>,
}

impl VsockCommon {
    /// `vhost_vsock_common_realize()`.
    fn realize(vdev: &mut VirtIODevice, name: &str) -> Result<()> {
        vdev.init(name, VIRTIO_ID_VSOCK, VIRTIO_VSOCK_CONFIG_SIZE);
        for _ in 0..3 {
            vdev.add_queue(VHOST_VSOCK_QUEUE_SIZE)?;
        }
        Ok(())
    }

    fn vhost(&self) -> Result<&VhostDev> {
        self.vhost.as_ref().ok_or_else(|| Error::generic("vhost-vsock: device is not realized"))
    }

    fn vhost_mut(&mut self) -> Result<&mut VhostDev> {
        self.vhost.as_mut().ok_or_else(|| Error::generic("vhost-vsock: device is not realized"))
    }

    /// `vhost_vsock_common_get_features()`.
    fn get_features(&self, features: u64) -> Result<u64> {
        let mut features = features;
        if self.seqpacket != OnOffAuto::Off {
            features |= feature(VIRTIO_VSOCK_F_SEQPACKET);
        }
        let features = self.vhost()?.get_features(FEATURE_BITS, features);
        if self.seqpacket == OnOffAuto::On && !has_feature(features, VIRTIO_VSOCK_F_SEQPACKET) {
            return Err(Error::generic("vhost-vsock backend doesn't support seqpacket"));
        }
        Ok(features)
    }

    fn is_started(&self) -> bool {
        self.vhost.as_ref().is_some_and(VhostDev::is_started)
    }

    /// `vhost_vsock_common_start()`.
    fn start(&mut self, vdev: &mut VirtIODevice) -> Result<()> {
        let mem = self.mem.clone();
        let vhost = self.vhost_mut()?;
        vhost.set_mem_table(mem)?;
        vhost.start(vdev, vdev.guest_features()).map_err(|e| e.prepend("Error starting vhost: "))
    }

    /// `vhost_vsock_common_stop()`.
    fn stop(&mut self, vdev: &mut VirtIODevice) -> Result<()> {
        self.vhost_mut()?.stop(vdev)
    }

    fn set_mem_table(&mut self, regions: Vec<VhostMemRegion>) -> Result<()> {
        self.mem = regions.clone();
        match self.vhost.as_mut() {
            Some(vhost) if vhost.is_started() => vhost.set_mem_table(regions),
            _ => Ok(()),
        }
    }

    fn handle_output(&self, queue: u16) {
        if let Some(vhost) = self.vhost.as_ref() {
            if vhost.is_started() && usize::from(queue) < vhost.nvqs() {
                if let Err(e) = vhost.kick(queue) {
                    error_report(&e.to_string());
                }
            }
        }
    }

    fn poll_calls(&self, vdev: &mut VirtIODevice) -> bool {
        self.vhost.as_ref().is_some_and(|v| v.poll_calls(vdev))
    }

    /// `vhost_vsock_common_send_transport_reset()`.
    fn send_transport_reset(vdev: &mut VirtIODevice) {
        let Some(chain) = vdev.pop(EVENT_QUEUE) else {
            error_report("vhost-vsock missed transport reset event");
            return;
        };
        if !chain.readable().is_empty() {
            error_report("invalid vhost-vsock event virtqueue element with out buffers");
            vdev.detach(EVENT_QUEUE, &chain);
            return;
        }
        let event = VIRTIO_VSOCK_EVENT_TRANSPORT_RESET.to_le_bytes();
        let mem = std::sync::Arc::clone(vdev.mem());
        let written = chain.writable_len() >= event.len() as u64
            && chain.writer(&*mem).write_all(&event).is_ok();
        if !written {
            error_report("vhost-vsock event virtqueue element is too short");
            vdev.detach(EVENT_QUEUE, &chain);
            return;
        }
        vdev.push(EVENT_QUEUE, &chain, event.len() as u32);
        vdev.notify(EVENT_QUEUE);
    }
}

/// Properties of [`VhostVsock`].
#[derive(Debug, Default)]
pub struct VhostVsockConf {
    /// `guest-cid`: the guest's address, above 2 and within 32 bits.
    pub guest_cid: u64,
    /// `vhostfd`: an already open `/dev/vhost-vsock`, instead of opening it.
    pub vhostfd: Option<File>,
    /// `seqpacket`: whether to offer `VIRTIO_VSOCK_F_SEQPACKET`.
    pub seqpacket: OnOffAuto,
}

/// `VHostVSock`: vsock through the kernel's vhost-vsock.
#[derive(Debug)]
pub struct VhostVsock {
    guest_cid: u64,
    vhostfd: Option<File>,
    common: VsockCommon,
}

impl VhostVsock {
    /// The device with properties `conf`.
    pub fn new(conf: VhostVsockConf) -> Self {
        VhostVsock {
            guest_cid: conf.guest_cid,
            vhostfd: conf.vhostfd,
            common: VsockCommon { seqpacket: conf.seqpacket, ..VsockCommon::default() },
        }
    }

    /// The guest CID.
    pub fn guest_cid(&self) -> u64 {
        self.guest_cid
    }

    /// The vhost device, once realized.
    pub fn vhost(&self) -> Option<&VhostDev> {
        self.common.vhost.as_ref()
    }

    /// Sets the guest memory the backend sees, see [`VhostDev::set_mem_table`].
    pub fn set_mem_table(&mut self, regions: Vec<VhostMemRegion>) -> Result<()> {
        self.common.set_mem_table(regions)
    }

    /// Interrupts the driver for every queue the backend signalled. Returns whether any was.
    pub fn poll_calls(&self, vdev: &mut VirtIODevice) -> bool {
        self.common.poll_calls(vdev)
    }

    /// `vhost_vsock_common_send_transport_reset()`: tells the driver every connection is gone,
    /// as QEMU does after migration.
    pub fn send_transport_reset(&self, vdev: &mut VirtIODevice) {
        VsockCommon::send_transport_reset(vdev);
    }

    #[cfg(target_os = "linux")]
    fn open(&mut self) -> Result<VhostKernel> {
        match self.vhostfd.take() {
            Some(file) => Ok(VhostKernel::from_file(file)),
            None => VhostKernel::open(VHOST_VSOCK_PATH).map_err(|e| {
                let msg = match e {
                    ruvm_vhost::Error::Io(e) => strerror(&e),
                    e => e.to_string(),
                };
                Error::generic(format!("Could not open '{VHOST_VSOCK_PATH}': {msg}"))
            }),
        }
    }

    #[cfg(target_os = "linux")]
    fn kernel(&mut self) -> Option<&mut VhostKernel> {
        match self.common.vhost.as_mut()?.connection() {
            VhostConnection::Kernel(k) => Some(k),
            VhostConnection::User(_) => None,
        }
    }

    /// `vhost_vsock_set_running()`.
    #[cfg(target_os = "linux")]
    fn set_running(&mut self, running: bool) -> Result<()> {
        match self.kernel() {
            Some(k) => k.vsock_set_running(running).map_err(|e| Error::generic(e.to_string())),
            None => Ok(()),
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn set_running(&mut self, _running: bool) -> Result<()> {
        Ok(())
    }
}

impl VirtioDeviceClass for VhostVsock {
    fn realize(&mut self, vdev: &mut VirtIODevice) -> Result<()> {
        if self.guest_cid <= 2 {
            return Err(Error::generic("guest-cid property must be greater than 2"));
        }
        if self.guest_cid > u64::from(u32::MAX) {
            return Err(Error::generic("guest-cid property must be a 32-bit number"));
        }
        #[cfg(target_os = "linux")]
        {
            let kernel = self.open()?;
            VsockCommon::realize(vdev, TYPE_VHOST_VSOCK)?;
            let vhost = VhostDev::new(
                VhostConnection::Kernel(kernel),
                VHOST_NVQS,
                VhostDevOptions::default(),
            )?;
            self.common.vhost = Some(vhost);
            let cid = self.guest_cid;
            let kernel = self.kernel().ok_or_else(|| Error::generic("vhost-vsock: no backend"))?;
            if let Err(e) = kernel.vsock_set_guest_cid(cid) {
                self.common.vhost = None;
                let msg = match e {
                    ruvm_vhost::Error::Io(e) => strerror(&e),
                    e => e.to_string(),
                };
                return Err(Error::generic(format!("vhost-vsock: unable to set guest cid: {msg}")));
            }
            Ok(())
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = vdev;
            let _ = self.vhostfd.take();
            // There is no vhost-vsock outside Linux: fail the way QEMU does when the device
            // file is missing.
            Err(Error::generic(format!(
                "Could not open '{VHOST_VSOCK_PATH}': No such file or directory"
            )))
        }
    }

    fn get_features(&mut self, _vdev: &VirtIODevice, features: u64) -> Result<u64> {
        self.common.get_features(features)
    }

    fn get_config(&mut self, _vdev: &VirtIODevice, config: &mut [u8]) {
        config[..8].copy_from_slice(&self.guest_cid.to_le_bytes());
    }

    fn set_status(&mut self, vdev: &mut VirtIODevice, status: u8) -> Result<()> {
        let should_start = status & VIRTIO_CONFIG_S_DRIVER_OK != 0;
        if self.common.vhost.is_none() || self.common.is_started() == should_start {
            return Ok(());
        }
        if should_start {
            if let Err(e) = self.common.start(vdev) {
                error_report(&e.to_string());
                return Ok(());
            }
            if let Err(e) = self.set_running(true) {
                let _ = self.common.stop(vdev);
                error_report(&format!("Error starting vhost vsock: {e}"));
            }
        } else {
            if let Err(e) = self.set_running(false) {
                error_report(&format!("vhost vsock set running failed: {e}"));
                return Ok(());
            }
            if let Err(e) = self.common.stop(vdev) {
                error_report(&e.to_string());
            }
        }
        Ok(())
    }

    fn handle_output(&mut self, _vdev: &mut VirtIODevice, queue: u16) {
        self.common.handle_output(queue);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Properties of [`VhostUserVsock`].
#[derive(Debug, Default)]
pub struct VhostUserVsockConf {
    /// `chardev`: the socket to the backend.
    pub chardev: Option<VhostUserChardev>,
    /// `seqpacket`: whether to offer `VIRTIO_VSOCK_F_SEQPACKET`.
    pub seqpacket: OnOffAuto,
}

/// `VHostUserVSock`: vsock through a vhost-user backend.
#[derive(Debug)]
pub struct VhostUserVsock {
    chardev: Option<VhostUserChardev>,
    config: [u8; VIRTIO_VSOCK_CONFIG_SIZE],
    common: VsockCommon,
}

impl VhostUserVsock {
    /// The device with properties `conf`.
    pub fn new(conf: VhostUserVsockConf) -> Self {
        VhostUserVsock {
            chardev: conf.chardev,
            config: [0; VIRTIO_VSOCK_CONFIG_SIZE],
            common: VsockCommon { seqpacket: conf.seqpacket, ..VsockCommon::default() },
        }
    }

    /// The guest CID the backend reported.
    pub fn guest_cid(&self) -> u64 {
        u64::from_le_bytes(self.config)
    }

    /// The vhost device, once realized.
    pub fn vhost(&self) -> Option<&VhostDev> {
        self.common.vhost.as_ref()
    }

    /// Sets the guest memory the backend sees, see [`VhostDev::set_mem_table`].
    pub fn set_mem_table(&mut self, regions: Vec<VhostMemRegion>) -> Result<()> {
        self.common.set_mem_table(regions)
    }

    /// Interrupts the driver for every queue the backend signalled. Returns whether any was.
    pub fn poll_calls(&self, vdev: &mut VirtIODevice) -> bool {
        self.common.poll_calls(vdev)
    }

    /// `vhost_vsock_common_send_transport_reset()`.
    pub fn send_transport_reset(&self, vdev: &mut VirtIODevice) {
        VsockCommon::send_transport_reset(vdev);
    }
}

impl VirtioDeviceClass for VhostUserVsock {
    fn realize(&mut self, vdev: &mut VirtIODevice) -> Result<()> {
        let Some(chardev) = self.chardev.take() else {
            return Err(Error::generic("missing chardev"));
        };
        let frontend = chardev.connect()?;
        VsockCommon::realize(vdev, TYPE_VHOST_USER_VSOCK)?;
        let mut vhost = VhostDev::new(
            VhostConnection::User(frontend),
            VHOST_NVQS,
            VhostDevOptions { supports_config: true },
        )?;
        let config = vhost.get_config(VIRTIO_VSOCK_CONFIG_SIZE)?;
        let n = config.len().min(VIRTIO_VSOCK_CONFIG_SIZE);
        self.config[..n].copy_from_slice(&config[..n]);
        self.common.vhost = Some(vhost);
        Ok(())
    }

    fn get_features(&mut self, _vdev: &VirtIODevice, features: u64) -> Result<u64> {
        let features = self.common.vhost()?.get_features(USER_FEATURE_BITS, features);
        self.common.get_features(features)
    }

    fn get_config(&mut self, _vdev: &VirtIODevice, config: &mut [u8]) {
        config[..VIRTIO_VSOCK_CONFIG_SIZE].copy_from_slice(&self.config);
    }

    fn set_status(&mut self, vdev: &mut VirtIODevice, status: u8) -> Result<()> {
        let should_start = status & VIRTIO_CONFIG_S_DRIVER_OK != 0;
        if self.common.vhost.is_none() || self.common.is_started() == should_start {
            return Ok(());
        }
        if should_start { self.common.start(vdev) } else { self.common.stop(vdev) }
    }

    fn handle_output(&mut self, _vdev: &mut VirtIODevice, queue: u16) {
        self.common.handle_output(queue);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
