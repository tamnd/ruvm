// SPDX-License-Identifier: GPL-2.0-or-later

//! The `virtio-balloon-device` description.

use ruvm_base::{Error, Result};

use super::{
    BALLOON_SVQ, FreePageHintStatus, VIRTIO_BALLOON_F_FREE_PAGE_HINT, VIRTIO_BALLOON_F_PAGE_POISON,
    VirtioBalloon,
};
use crate::virtio::{VIRTIO_CONFIG_S_DRIVER_OK, VirtIODevice};

/// `VirtIOBalloon` as `virtio-balloon-device` version 1 has it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VirtioBalloonVmState {
    pub num_pages: u32,
    pub actual: u32,
    /// `free_page_hint_cmd_id`, in `virtio-balloon-device/free-page-report` when the device
    /// offers `VIRTIO_BALLOON_F_FREE_PAGE_HINT`.
    pub free_page_hint_cmd_id: u32,
    /// `free_page_hint_status` as QEMU numbers it: stop 0, requested 1, start 2, done 3.
    pub free_page_hint_status: u32,
    /// `poison_val`, in `virtio-balloon-device/page-poison` when the driver accepted
    /// `VIRTIO_BALLOON_F_PAGE_POISON`.
    pub poison_val: u32,
    /// Whether the free page hint subsection is sent. Not migrated.
    pub free_page_hint: bool,
    /// Whether the page poison subsection is sent. Not migrated.
    pub page_poison: bool,
}

impl VirtioBalloon {
    /// The device model's part of the migration stream.
    pub fn vmstate_save(&self, vdev: &VirtIODevice) -> VirtioBalloonVmState {
        VirtioBalloonVmState {
            num_pages: self.num_pages,
            actual: self.actual,
            free_page_hint_cmd_id: self.free_page_hint_cmd_id,
            free_page_hint_status: match self.free_page_hint_status {
                FreePageHintStatus::Stop => 0,
                FreePageHintStatus::Requested => 1,
                FreePageHintStatus::Start => 2,
                FreePageHintStatus::Done => 3,
            },
            poison_val: self.poison_val,
            free_page_hint: vdev.has_feature(VIRTIO_BALLOON_F_FREE_PAGE_HINT),
            page_poison: vdev.has_feature(VIRTIO_BALLOON_F_PAGE_POISON),
        }
    }

    /// Takes back what [`vmstate_save`](Self::vmstate_save) returned.
    pub fn vmstate_load(&mut self, s: &VirtioBalloonVmState) -> Result<()> {
        self.num_pages = s.num_pages;
        self.actual = s.actual;
        self.free_page_hint_cmd_id = s.free_page_hint_cmd_id;
        self.free_page_hint_status = match s.free_page_hint_status {
            0 => FreePageHintStatus::Stop,
            1 => FreePageHintStatus::Requested,
            2 => FreePageHintStatus::Start,
            3 => FreePageHintStatus::Done,
            v => {
                return Err(Error::generic(format!(
                    "virtio-balloon: bad free page hint status {v}"
                )));
            }
        };
        self.poison_val = s.poison_val;
        Ok(())
    }

    /// `virtio_balloon_post_load_device()` and the rewind `virtio_balloon_set_status()` does
    /// when the VM runs: the statistics buffer the guest gave the source is not migrated, so
    /// it is taken from the queue again.
    pub(super) fn vmstate_post_load(&mut self, vdev: &mut VirtIODevice) -> Result<()> {
        if self.stats_elem.is_none()
            && vdev.status() & VIRTIO_CONFIG_S_DRIVER_OK != 0
            && vdev.rewind(BALLOON_SVQ, 1)
        {
            self.receive_stats(vdev);
        }
        if self.stats_enabled() {
            self.stats_timer = Some(self.stats_poll_interval as u64);
        }
        Ok(())
    }
}
