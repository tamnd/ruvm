// SPDX-License-Identifier: GPL-2.0-or-later

//! vhost-user-fs, a port of `hw/virtio/vhost-user-fs.c`.
//!
//! virtio-fs shares a host directory with the guest. All the work happens in a vhost-user
//! backend such as virtiofsd: the device only offers the queues, one high priority queue and
//! `num-request-queues` request queues, and a config space with the mount tag and the number
//! of request queues.
//!
//! Differences from QEMU:
//!
//! - There is no ioeventfd or irqfd. A guest kick reaches `handle_output`, which passes it to
//!   the backend, and the owner of the device calls [`VhostUserFs::poll_calls`] when the
//!   backend signals. See the [`vhost`](crate::vhost) module.
//! - The guest memory table comes from the caller through [`VhostUserFs::set_mem_table`].
//! - Migration, the DAX window and the `bootindex` property are not ported.

use std::any::Any;

use ruvm_base::{Error, Result, error_report};
use ruvm_virtio_queue::VIRTIO_F_RING_PACKED;

use crate::vhost::{VhostConnection, VhostDev, VhostDevOptions, VhostMemRegion, VhostUserChardev};
use crate::virtio::{
    VIRTIO_CONFIG_S_DRIVER_OK, VIRTIO_F_IN_ORDER, VIRTIO_F_IOMMU_PLATFORM,
    VIRTIO_F_NOTIFICATION_DATA, VIRTIO_F_NOTIFY_ON_EMPTY, VIRTIO_F_RING_RESET, VIRTIO_F_VERSION_1,
    VirtIODevice, VirtioDeviceClass,
};

/// `TYPE_VHOST_USER_FS`.
pub const TYPE_VHOST_USER_FS: &str = "vhost-user-fs-device";
/// `VIRTIO_ID_FS`.
pub const VIRTIO_ID_FS: u16 = 26;
/// The size of `virtio_fs_config.tag`.
pub const VIRTIO_FS_TAG_SIZE: usize = 36;
/// The size of `struct virtio_fs_config`.
pub const VIRTIO_FS_CONFIG_SIZE: usize = 40;
/// `VIRTQUEUE_MAX_SIZE`.
const VIRTQUEUE_MAX_SIZE: u16 = 1024;

const VIRTIO_RING_F_INDIRECT_DESC: u32 = 28;
const VIRTIO_RING_F_EVENT_IDX: u32 = 29;

/// `user_feature_bits` in `vhost-user-fs.c`.
const USER_FEATURE_BITS: &[u32] = &[
    VIRTIO_F_VERSION_1,
    VIRTIO_RING_F_INDIRECT_DESC,
    VIRTIO_RING_F_EVENT_IDX,
    VIRTIO_F_NOTIFY_ON_EMPTY,
    VIRTIO_F_RING_PACKED,
    VIRTIO_F_IOMMU_PLATFORM,
    VIRTIO_F_RING_RESET,
    VIRTIO_F_IN_ORDER,
    VIRTIO_F_NOTIFICATION_DATA,
];

/// Properties of [`VhostUserFs`], `VHostUserFSConf`.
#[derive(Debug)]
pub struct VhostUserFsConf {
    /// `chardev`: the socket to the backend.
    pub chardev: Option<VhostUserChardev>,
    /// `tag`: the name the guest mounts, 1 to 36 bytes.
    pub tag: Option<String>,
    /// `num-request-queues`, default 1.
    pub num_request_queues: u16,
    /// `queue-size`, default 128.
    pub queue_size: u16,
}

impl Default for VhostUserFsConf {
    fn default() -> Self {
        VhostUserFsConf { chardev: None, tag: None, num_request_queues: 1, queue_size: 128 }
    }
}

/// `VHostUserFS`.
#[derive(Debug)]
pub struct VhostUserFs {
    chardev: Option<VhostUserChardev>,
    tag: Option<String>,
    num_request_queues: u16,
    queue_size: u16,
    vhost: Option<VhostDev>,
    mem: Vec<VhostMemRegion>,
}

impl VhostUserFs {
    /// The device with properties `conf`.
    pub fn new(conf: VhostUserFsConf) -> Self {
        VhostUserFs {
            chardev: conf.chardev,
            tag: conf.tag,
            num_request_queues: conf.num_request_queues,
            queue_size: conf.queue_size,
            vhost: None,
            mem: Vec::new(),
        }
    }

    /// The mount tag.
    pub fn tag(&self) -> Option<&str> {
        self.tag.as_deref()
    }

    /// The number of request queues.
    pub fn num_request_queues(&self) -> u16 {
        self.num_request_queues
    }

    /// The vhost device, once realized.
    pub fn vhost(&self) -> Option<&VhostDev> {
        self.vhost.as_ref()
    }

    /// Sets the guest memory the backend sees, see [`VhostDev::set_mem_table`].
    pub fn set_mem_table(&mut self, regions: Vec<VhostMemRegion>) -> Result<()> {
        self.mem = regions.clone();
        match self.vhost.as_mut() {
            Some(vhost) if vhost.is_started() => vhost.set_mem_table(regions),
            _ => Ok(()),
        }
    }

    /// Interrupts the driver for every queue the backend signalled. Returns whether any was.
    pub fn poll_calls(&self, vdev: &mut VirtIODevice) -> bool {
        self.vhost.as_ref().is_some_and(|v| v.poll_calls(vdev))
    }

    fn check(&self) -> Result<()> {
        if self.chardev.is_none() {
            return Err(Error::generic("missing chardev"));
        }
        let Some(tag) = self.tag.as_deref() else {
            return Err(Error::generic("missing tag property"));
        };
        if tag.is_empty() {
            return Err(Error::generic("tag property cannot be empty"));
        }
        if tag.len() > VIRTIO_FS_TAG_SIZE {
            return Err(Error::generic(format!(
                "tag property must be {VIRTIO_FS_TAG_SIZE} bytes or less"
            )));
        }
        if self.num_request_queues == 0 {
            return Err(Error::generic("num-request-queues property must be larger than 0"));
        }
        if !self.queue_size.is_power_of_two() {
            return Err(Error::generic("queue-size property must be a power of 2"));
        }
        if self.queue_size > VIRTQUEUE_MAX_SIZE {
            return Err(Error::generic(format!(
                "queue-size property must be {VIRTQUEUE_MAX_SIZE} or smaller"
            )));
        }
        Ok(())
    }
}

impl VirtioDeviceClass for VhostUserFs {
    fn realize(&mut self, vdev: &mut VirtIODevice) -> Result<()> {
        self.check()?;
        let Some(chardev) = self.chardev.take() else {
            return Err(Error::generic("missing chardev"));
        };
        let frontend = chardev.connect()?;
        vdev.init(TYPE_VHOST_USER_FS, VIRTIO_ID_FS, VIRTIO_FS_CONFIG_SIZE);
        for _ in 0..=self.num_request_queues {
            vdev.add_queue(self.queue_size)?;
        }
        let nvqs = 1 + usize::from(self.num_request_queues);
        let vhost =
            VhostDev::new(VhostConnection::User(frontend), nvqs, VhostDevOptions::default())?;
        self.vhost = Some(vhost);
        Ok(())
    }

    fn get_features(&mut self, _vdev: &VirtIODevice, features: u64) -> Result<u64> {
        match self.vhost.as_ref() {
            Some(vhost) => Ok(vhost.get_features(USER_FEATURE_BITS, features)),
            None => Ok(features),
        }
    }

    fn get_config(&mut self, _vdev: &VirtIODevice, config: &mut [u8]) {
        let mut fscfg = [0u8; VIRTIO_FS_CONFIG_SIZE];
        let tag = self.tag.as_deref().unwrap_or("").as_bytes();
        // The tag with its NUL terminator, cut at 36 bytes.
        let n = tag.len().min(VIRTIO_FS_TAG_SIZE);
        fscfg[..n].copy_from_slice(&tag[..n]);
        fscfg[VIRTIO_FS_TAG_SIZE..]
            .copy_from_slice(&u32::from(self.num_request_queues).to_le_bytes());
        config[..VIRTIO_FS_CONFIG_SIZE].copy_from_slice(&fscfg);
    }

    fn set_status(&mut self, vdev: &mut VirtIODevice, status: u8) -> Result<()> {
        let should_start = status & VIRTIO_CONFIG_S_DRIVER_OK != 0;
        let mem = self.mem.clone();
        let Some(vhost) = self.vhost.as_mut() else {
            return Ok(());
        };
        if vhost.is_started() == should_start {
            return Ok(());
        }
        if should_start {
            // vuf_start() reports the failure and carries on.
            let started =
                vhost.set_mem_table(mem).and_then(|()| vhost.start(vdev, vdev.guest_features()));
            if let Err(e) = started {
                error_report(&format!("Error starting vhost: {e}"));
            }
            Ok(())
        } else {
            vhost.stop(vdev)
        }
    }

    fn handle_output(&mut self, _vdev: &mut VirtIODevice, queue: u16) {
        if let Some(vhost) = self.vhost.as_ref() {
            if vhost.is_started() {
                if let Err(e) = vhost.kick(queue) {
                    error_report(&e.to_string());
                }
            }
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
