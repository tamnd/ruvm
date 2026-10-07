// SPDX-License-Identifier: GPL-2.0-or-later

//! virtio-balloon, a port of `hw/virtio/virtio-balloon.c`.
//!
//! The host asks the guest to give back memory by raising `num_pages` in config space. The
//! guest answers by sending page frame numbers on the inflate queue, and takes pages back with
//! the deflate queue. What happens to those pages is up to a [`BalloonBackend`]: QEMU discards
//! inflated pages from its RAM blocks and prefaults deflated ones. The statistics queue carries
//! memory statistics from the guest, the free page hint queue lets migration skip pages the
//! guest does not use, and the reporting queue hands free pages back for discarding.
//!
//! The QMP `balloon` and `query-balloon` commands are [`VirtioBalloon::to_target`] and
//! [`VirtioBalloon::query`]. The `guest-stats` and `guest-stats-polling-interval` properties
//! are [`VirtioBalloon::guest_stats`] and [`VirtioBalloon::set_stats_poll_interval`].
//!
//! Differences from QEMU:
//!
//! - The statistics timer is not a real timer. [`VirtioBalloon::stats_timer`] says whether it
//!   is armed and with what delay, and whoever owns the device calls
//!   [`VirtioBalloon::stats_poll`] when it expires.
//! - Free page hints are handled right away when the driver kicks, and the device stops looking
//!   at the queue once it is empty. QEMU runs this in an iothread bottom half that keeps polling
//!   the queue while hinting is in progress. `free-page-hint` does not need an `iothread`.
//! - The migration notifier that starts and stops hinting is not wired. Call
//!   [`VirtioBalloon::free_page_start`], [`VirtioBalloon::free_page_stop`] and
//!   [`VirtioBalloon::free_page_done`] instead.
//! - `BALLOON_CHANGE` events are collected in a list, see
//!   [`VirtioBalloon::take_balloon_change_events`].
//!
//! Migration state is in the `vmstate` submodule, [`VirtioBalloonVmState`]. The statistics
//! buffer the guest handed over is not migrated; the destination takes it from the queue
//! again after loading, as QEMU's `set_status` rewind does when the VM runs.
//!
//! Not ported: trace points, QOM registration, the single balloon handler registry, the
//! rewind of `set_status` for a VM stopped and resumed without migrating, and the `qom-get`
//! visitor for `guest-stats`.

use std::any::Any;
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{SystemTime, UNIX_EPOCH};

use ruvm_base::{Error, Result};

use crate::virtio::{VIRTQUEUE_MAX_SIZE, VirtIODevice, VirtioDeviceClass, feature};

mod vmstate;

pub use vmstate::VirtioBalloonVmState;

/// `TYPE_VIRTIO_BALLOON`.
pub const TYPE_VIRTIO_BALLOON: &str = "virtio-balloon-device";

/// `VIRTIO_ID_BALLOON`.
pub const VIRTIO_ID_BALLOON: u16 = 5;

/// `VIRTIO_BALLOON_F_MUST_TELL_HOST`.
pub const VIRTIO_BALLOON_F_MUST_TELL_HOST: u32 = 0;
/// `VIRTIO_BALLOON_F_STATS_VQ`: there is a statistics queue.
pub const VIRTIO_BALLOON_F_STATS_VQ: u32 = 1;
/// `VIRTIO_BALLOON_F_DEFLATE_ON_OOM`: the guest may deflate when it runs out of memory.
pub const VIRTIO_BALLOON_F_DEFLATE_ON_OOM: u32 = 2;
/// `VIRTIO_BALLOON_F_FREE_PAGE_HINT`: there is a free page hint queue.
pub const VIRTIO_BALLOON_F_FREE_PAGE_HINT: u32 = 3;
/// `VIRTIO_BALLOON_F_PAGE_POISON`: the guest tells the host what free pages are filled with.
pub const VIRTIO_BALLOON_F_PAGE_POISON: u32 = 4;
/// `VIRTIO_BALLOON_F_REPORTING`: there is a free page reporting queue.
pub const VIRTIO_BALLOON_F_REPORTING: u32 = 5;

/// `VIRTIO_BALLOON_PFN_SHIFT`: page frame numbers count 4 KiB pages.
pub const VIRTIO_BALLOON_PFN_SHIFT: u32 = 12;
/// `BALLOON_PAGE_SIZE`.
pub const BALLOON_PAGE_SIZE: u64 = 1 << VIRTIO_BALLOON_PFN_SHIFT;

/// `VIRTIO_BALLOON_CMD_ID_STOP`.
pub const VIRTIO_BALLOON_CMD_ID_STOP: u32 = 0;
/// `VIRTIO_BALLOON_CMD_ID_DONE`.
pub const VIRTIO_BALLOON_CMD_ID_DONE: u32 = 1;
/// `VIRTIO_BALLOON_FREE_PAGE_HINT_CMD_ID_MIN`: the first command id, and where they wrap to.
pub const VIRTIO_BALLOON_FREE_PAGE_HINT_CMD_ID_MIN: u32 = 0x8000_0000;

/// `VIRTIO_BALLOON_S_NR`: how many statistics there are.
pub const VIRTIO_BALLOON_S_NR: usize = 16;

/// `balloon_stat_names`: the name of each statistic, indexed by tag.
pub const STAT_NAMES: [&str; VIRTIO_BALLOON_S_NR] = [
    "stat-swap-in",
    "stat-swap-out",
    "stat-major-faults",
    "stat-minor-faults",
    "stat-free-memory",
    "stat-total-memory",
    "stat-available-memory",
    "stat-disk-caches",
    "stat-htlb-pgalloc",
    "stat-htlb-pgfail",
    "stat-oom-kills",
    "stat-alloc-stalls",
    "stat-async-scans",
    "stat-direct-scans",
    "stat-async-reclaims",
    "stat-direct-reclaims",
];

/// The size of one `virtio_balloon_stat` entry: a u16 tag and a u64 value, packed.
const STAT_ENTRY_SIZE: usize = 10;

/// The queue sizes.
const BALLOON_QUEUE_SIZE: u16 = 128;
const REPORTING_QUEUE_SIZE: u16 = 32;

/// Config space offsets, `struct virtio_balloon_config`.
const CFG_NUM_PAGES: usize = 0;
const CFG_ACTUAL: usize = 4;
const CFG_FREE_PAGE_HINT_CMD_ID: usize = 8;
const CFG_POISON_VAL: usize = 12;
/// `sizeof(struct virtio_balloon_config)` as far as QEMU fills it.
const CONFIG_SIZE: usize = 16;

/// What the balloon does to guest memory, standing in for the RAM block calls QEMU makes.
///
/// Only [`discard`](Self::discard) and [`populate`](Self::populate) have to be written.
pub trait BalloonBackend: Send + fmt::Debug {
    /// Whether the 4 KiB page at `gpa` is RAM. Page numbers that point anywhere else are
    /// ignored.
    fn is_ram(&self, _gpa: u64) -> bool {
        true
    }

    /// The host page size backing `gpa`. When it is bigger than 4 KiB, inflated pages are
    /// collected until a whole host page is in the balloon.
    fn host_page_size(&self, _gpa: u64) -> u64 {
        BALLOON_PAGE_SIZE
    }

    /// `virtio_balloon_inhibited()`: whether discarding RAM is off, for example because a VFIO
    /// device pins it or postcopy is running. Inflate, deflate and reporting then do nothing.
    fn discard_inhibited(&self) -> bool {
        false
    }

    /// `ram_block_discard_range()`: the guest gave up `len` bytes at `gpa`.
    fn discard(&mut self, gpa: u64, len: u64);

    /// The `QEMU_MADV_WILLNEED` hint on deflate: the guest is about to use `len` bytes at `gpa`
    /// again.
    fn populate(&mut self, gpa: u64, len: u64);

    /// `qemu_guest_free_page_hint()`: the guest says `len` bytes at `gpa` are free, so
    /// migration does not have to send them.
    fn free_page_hint(&mut self, _gpa: u64, _len: u64) {}
}

/// One call a [`RecordingBalloonBackend`] saw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BalloonOp {
    /// [`BalloonBackend::discard`].
    Discard(u64, u64),
    /// [`BalloonBackend::populate`].
    Populate(u64, u64),
    /// [`BalloonBackend::free_page_hint`].
    FreePageHint(u64, u64),
}

/// A backend that only writes down what it is asked to do. Clones share the list, so a test
/// can keep one and give the other to the device.
#[derive(Clone, Debug)]
pub struct RecordingBalloonBackend {
    ops: Arc<Mutex<Vec<BalloonOp>>>,
    ram_size: Option<u64>,
    host_page_size: u64,
    inhibited: bool,
}

impl Default for RecordingBalloonBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl RecordingBalloonBackend {
    /// A backend where every address is RAM and host pages are 4 KiB.
    pub fn new() -> Self {
        RecordingBalloonBackend {
            ops: Arc::new(Mutex::new(Vec::new())),
            ram_size: None,
            host_page_size: BALLOON_PAGE_SIZE,
            inhibited: false,
        }
    }

    /// Only addresses below `ram_size` are RAM.
    pub fn with_ram_size(mut self, ram_size: u64) -> Self {
        self.ram_size = Some(ram_size);
        self
    }

    /// RAM is backed by pages of `size` bytes.
    pub fn with_host_page_size(mut self, size: u64) -> Self {
        self.host_page_size = size;
        self
    }

    /// Discarding RAM is inhibited.
    pub fn with_discard_inhibited(mut self, inhibited: bool) -> Self {
        self.inhibited = inhibited;
        self
    }

    /// Everything recorded so far.
    pub fn ops(&self) -> Vec<BalloonOp> {
        self.ops.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    /// Takes everything recorded so far, leaving the list empty.
    pub fn take_ops(&self) -> Vec<BalloonOp> {
        std::mem::take(&mut *self.ops.lock().unwrap_or_else(PoisonError::into_inner))
    }

    fn record(&self, op: BalloonOp) {
        self.ops.lock().unwrap_or_else(PoisonError::into_inner).push(op);
    }
}

impl BalloonBackend for RecordingBalloonBackend {
    fn is_ram(&self, gpa: u64) -> bool {
        self.ram_size
            .is_none_or(|size| gpa.checked_add(BALLOON_PAGE_SIZE).is_some_and(|e| e <= size))
    }

    fn host_page_size(&self, _gpa: u64) -> u64 {
        self.host_page_size
    }

    fn discard_inhibited(&self) -> bool {
        self.inhibited
    }

    fn discard(&mut self, gpa: u64, len: u64) {
        self.record(BalloonOp::Discard(gpa, len));
    }

    fn populate(&mut self, gpa: u64, len: u64) {
        self.record(BalloonOp::Populate(gpa, len));
    }

    fn free_page_hint(&mut self, gpa: u64, len: u64) {
        self.record(BalloonOp::FreePageHint(gpa, len));
    }
}

/// `FreePageHintStatus`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FreePageHintStatus {
    /// No hinting, or the guest was told to stop.
    #[default]
    Stop,
    /// The host asked for hints and waits for the guest to start.
    Requested,
    /// The guest is sending hints.
    Start,
    /// Hinting is over and the guest may reuse the pages.
    Done,
}

/// `PartiallyBalloonedPage`: the 4 KiB pieces of one big host page inflated so far.
#[derive(Debug, Default)]
struct PartialPage {
    base: u64,
    bitmap: Vec<bool>,
}

/// The virtio-balloon device model, `VirtIOBalloon`.
#[derive(Debug)]
pub struct VirtioBalloon {
    backend: Box<dyn BalloonBackend>,
    ram_size: u64,
    host_features: u64,
    num_pages: u32,
    actual: u32,
    poison_val: u32,
    stats: [u64; VIRTIO_BALLOON_S_NR],
    stats_last_update: i64,
    stats_poll_interval: i64,
    stats_timer: Option<u64>,
    stats_elem: Option<ruvm_virtio_queue::DescriptorChain>,
    free_page_hint_cmd_id: u32,
    free_page_hint_status: FreePageHintStatus,
    free_page_vq: Option<u16>,
    reporting_vq: Option<u16>,
    change_events: Vec<u64>,
}

/// The inflate queue.
pub const BALLOON_IVQ: u16 = 0;
/// The deflate queue.
pub const BALLOON_DVQ: u16 = 1;
/// The statistics queue.
pub const BALLOON_SVQ: u16 = 2;

impl VirtioBalloon {
    /// A balloon for a guest with `ram_size` bytes of RAM, with QEMU's default properties:
    /// only `page-poison` is on.
    pub fn new(ram_size: u64, backend: Box<dyn BalloonBackend>) -> Self {
        VirtioBalloon {
            backend,
            ram_size,
            host_features: feature(VIRTIO_BALLOON_F_PAGE_POISON),
            num_pages: 0,
            actual: 0,
            poison_val: 0,
            stats: [u64::MAX; VIRTIO_BALLOON_S_NR],
            stats_last_update: 0,
            stats_poll_interval: 0,
            stats_timer: None,
            stats_elem: None,
            free_page_hint_cmd_id: VIRTIO_BALLOON_FREE_PAGE_HINT_CMD_ID_MIN,
            free_page_hint_status: FreePageHintStatus::Stop,
            free_page_vq: None,
            reporting_vq: None,
            change_events: Vec::new(),
        }
    }

    fn set_property(&mut self, bit: u32, on: bool) {
        if on {
            self.host_features |= feature(bit);
        } else {
            self.host_features &= !feature(bit);
        }
    }

    /// The `deflate-on-oom` property.
    pub fn set_deflate_on_oom(&mut self, on: bool) {
        self.set_property(VIRTIO_BALLOON_F_DEFLATE_ON_OOM, on);
    }

    /// The `free-page-hint` property.
    pub fn set_free_page_hint(&mut self, on: bool) {
        self.set_property(VIRTIO_BALLOON_F_FREE_PAGE_HINT, on);
    }

    /// The `page-poison` property.
    pub fn set_page_poison(&mut self, on: bool) {
        self.set_property(VIRTIO_BALLOON_F_PAGE_POISON, on);
    }

    /// The `free-page-reporting` property.
    pub fn set_free_page_reporting(&mut self, on: bool) {
        self.set_property(VIRTIO_BALLOON_F_REPORTING, on);
    }

    /// The feature properties as a mask.
    pub fn host_features(&self) -> u64 {
        self.host_features
    }

    /// The backend.
    pub fn backend(&self) -> &dyn BalloonBackend {
        &*self.backend
    }

    /// How many pages the host wants in the balloon.
    pub fn num_pages(&self) -> u32 {
        self.num_pages
    }

    /// How many pages the guest says are in the balloon.
    pub fn actual(&self) -> u32 {
        self.actual
    }

    /// What the guest fills free pages with.
    pub fn poison_val(&self) -> u32 {
        self.poison_val
    }

    /// The free page hint command id.
    pub fn free_page_hint_cmd_id(&self) -> u32 {
        self.free_page_hint_cmd_id
    }

    /// Where free page hinting is at.
    pub fn free_page_hint_status(&self) -> FreePageHintStatus {
        self.free_page_hint_status
    }

    /// The free page hint queue, if the device has one.
    pub fn free_page_vq(&self) -> Option<u16> {
        self.free_page_vq
    }

    /// The free page reporting queue, if the device has one.
    pub fn reporting_vq(&self) -> Option<u16> {
        self.reporting_vq
    }

    /// Takes the `BALLOON_CHANGE` events sent so far. Each is the guest memory size in bytes.
    pub fn take_balloon_change_events(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.change_events)
    }

    // QMP.

    /// `virtio_balloon_to_target()`, the QMP `balloon` command: asks the guest to shrink to
    /// `target` bytes. A target of 0 is ignored.
    pub fn to_target(&mut self, vdev: &mut VirtIODevice, target: u64) {
        let target = target.min(self.ram_size);
        if target != 0 {
            self.num_pages = ((self.ram_size - target) >> VIRTIO_BALLOON_PFN_SHIFT) as u32;
            vdev.notify_config();
        }
    }

    /// `virtio_balloon_stat()`, the QMP `query-balloon` command: the guest memory size in bytes
    /// after taking out the balloon.
    pub fn query(&self) -> u64 {
        self.ram_size.saturating_sub(u64::from(self.actual) << VIRTIO_BALLOON_PFN_SHIFT)
    }

    // Statistics.

    /// The `guest-stats` property: every statistic by name. Those the guest never sent are
    /// `u64::MAX`, which QEMU shows as -1.
    pub fn guest_stats(&self) -> Vec<(&'static str, u64)> {
        STAT_NAMES.iter().copied().zip(self.stats).collect()
    }

    /// When the guest last sent statistics, in seconds since the epoch.
    pub fn stats_last_update(&self) -> i64 {
        self.stats_last_update
    }

    /// The `guest-stats-polling-interval` property, in seconds.
    pub fn stats_poll_interval(&self) -> i64 {
        self.stats_poll_interval
    }

    /// The delay in seconds the statistics timer was last armed with, or `None` when it is not
    /// armed.
    pub fn stats_timer(&self) -> Option<u64> {
        self.stats_timer
    }

    fn stats_enabled(&self) -> bool {
        self.stats_poll_interval > 0
    }

    /// `balloon_stats_set_poll_interval()`: sets the statistics polling interval. 0 turns
    /// polling off. Turning it on arms the timer to fire right away.
    pub fn set_stats_poll_interval(&mut self, value: i64) -> Result<()> {
        if value < 0 {
            return Err(Error::generic("timer value must be greater than zero"));
        }
        if value > i64::from(u32::MAX) {
            return Err(Error::generic("timer value is too big"));
        }
        if value == self.stats_poll_interval {
            return Ok(());
        }
        if value == 0 {
            self.stats_timer = None;
            self.stats_poll_interval = 0;
            return Ok(());
        }
        let was_enabled = self.stats_enabled();
        self.stats_poll_interval = value;
        self.stats_timer = Some(if was_enabled { value as u64 } else { 0 });
        Ok(())
    }

    /// `balloon_stats_poll_cb()`: the statistics timer expired. Gives the held statistics
    /// buffer back so the guest refills it. Does nothing if the timer is not armed.
    pub fn stats_poll(&mut self, vdev: &mut VirtIODevice) {
        if self.stats_timer.take().is_none() {
            return;
        }
        let supported = vdev.has_feature(VIRTIO_BALLOON_F_STATS_VQ);
        match self.stats_elem.take() {
            Some(elem) if supported => {
                vdev.push(BALLOON_SVQ, &elem, 0);
                vdev.notify(BALLOON_SVQ);
            }
            elem => {
                self.stats_elem = elem;
                self.stats_timer = Some(self.stats_poll_interval as u64);
            }
        }
    }

    /// `virtio_balloon_receive_stats()`.
    fn receive_stats(&mut self, vdev: &mut VirtIODevice) {
        if let Some(elem) = vdev.pop(BALLOON_SVQ) {
            if let Some(old) = self.stats_elem.take() {
                // This should never happen if the driver follows the spec.
                vdev.push(BALLOON_SVQ, &old, 0);
                vdev.notify(BALLOON_SVQ);
            }
            self.stats = [u64::MAX; VIRTIO_BALLOON_S_NR];
            let mem = Arc::clone(vdev.mem());
            let out = elem.reader(&*mem).read_to_vec().unwrap_or_default();
            for entry in out.chunks_exact(STAT_ENTRY_SIZE) {
                let tag = usize::from(u16::from_le_bytes([entry[0], entry[1]]));
                let mut val = [0; 8];
                val.copy_from_slice(&entry[2..]);
                if tag < VIRTIO_BALLOON_S_NR {
                    self.stats[tag] = u64::from_le_bytes(val);
                }
            }
            self.stats_elem = Some(elem);
            self.stats_last_update =
                SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64);
        }
        if self.stats_enabled() {
            self.stats_timer = Some(self.stats_poll_interval as u64);
        }
    }

    // Inflate and deflate.

    /// `balloon_inflate_page()`.
    fn inflate_page(&mut self, pa: u64, partial: &mut Option<PartialPage>) {
        let page_size = self.backend.host_page_size(pa).max(BALLOON_PAGE_SIZE);
        if page_size == BALLOON_PAGE_SIZE {
            self.backend.discard(pa, BALLOON_PAGE_SIZE);
            return;
        }
        // A piece of a bigger host page. Keep track of the pieces until the whole page is in
        // the balloon, then discard it. A piece of another page makes us forget the old one.
        let base = pa & !(page_size - 1);
        let subpages = (page_size / BALLOON_PAGE_SIZE) as usize;
        if partial.as_ref().is_some_and(|p| p.base != base || p.bitmap.len() != subpages) {
            *partial = None;
        }
        let p = partial.get_or_insert_with(|| PartialPage { base, bitmap: vec![false; subpages] });
        p.bitmap[((pa - base) / BALLOON_PAGE_SIZE) as usize] = true;
        if p.bitmap.iter().all(|&b| b) {
            self.backend.discard(base, page_size);
            *partial = None;
        }
    }

    /// `balloon_deflate_page()`.
    fn deflate_page(&mut self, pa: u64) {
        let page_size = self.backend.host_page_size(pa).max(BALLOON_PAGE_SIZE);
        self.backend.populate(pa & !(page_size - 1), page_size);
    }

    /// `virtio_balloon_handle_output()` for the inflate and deflate queues.
    fn handle_output_pages(&mut self, vdev: &mut VirtIODevice, queue: u16) {
        let mem = Arc::clone(vdev.mem());
        while let Some(elem) = vdev.pop(queue) {
            let mut partial = None;
            let out = elem.reader(&*mem).read_to_vec().unwrap_or_default();
            for pfn in out.chunks_exact(4) {
                let pfn = u32::from_le_bytes([pfn[0], pfn[1], pfn[2], pfn[3]]);
                let pa = u64::from(pfn) << VIRTIO_BALLOON_PFN_SHIFT;
                if !self.backend.is_ram(pa) || self.backend.discard_inhibited() {
                    continue;
                }
                if queue == BALLOON_IVQ {
                    self.inflate_page(pa, &mut partial);
                } else {
                    self.deflate_page(pa);
                }
            }
            vdev.push(queue, &elem, 0);
            vdev.notify(queue);
        }
    }

    // Free page hints.

    /// `get_free_page_hints()`: handles one buffer. Returns whether there was one.
    fn get_free_page_hints(&mut self, vdev: &mut VirtIODevice, vq: u16) -> bool {
        let Some(elem) = vdev.pop(vq) else {
            return false;
        };
        let mem = Arc::clone(vdev.mem());
        if !elem.readable().is_empty() {
            let mut id = [0; 4];
            let mut r = elem.reader(&*mem);
            if r.read_exact(&mut id).is_err() {
                vdev.error("received an incorrect cmd id");
                vdev.push(vq, &elem, 0);
                return false;
            }
            let id = u32::from_le_bytes(id);
            match self.free_page_hint_status {
                FreePageHintStatus::Requested if id == self.free_page_hint_cmd_id => {
                    self.free_page_hint_status = FreePageHintStatus::Start;
                }
                FreePageHintStatus::Start => {
                    // Only stop once hinting started, so a stale stop for the previous command
                    // does nothing.
                    self.free_page_hint_status = FreePageHintStatus::Stop;
                }
                _ => {}
            }
        }
        if self.free_page_hint_status == FreePageHintStatus::Start {
            for d in elem.writable() {
                self.backend.free_page_hint(d.addr(), u64::from(d.len()));
            }
        }
        vdev.push(vq, &elem, 0);
        true
    }

    /// `virtio_ballloon_get_free_page_hints()`.
    fn handle_free_page_vq(&mut self, vdev: &mut VirtIODevice, vq: u16) {
        loop {
            vdev.set_queue_notification(vq, false);
            let more = self.get_free_page_hints(vdev, vq);
            vdev.notify(vq);
            if !more {
                break;
            }
        }
        vdev.set_queue_notification(vq, true);
    }

    /// `virtio_balloon_free_page_start()`: asks the guest for free page hints with a new
    /// command id.
    pub fn free_page_start(&mut self, vdev: &mut VirtIODevice) {
        self.free_page_hint_cmd_id = if self.free_page_hint_cmd_id == u32::MAX {
            VIRTIO_BALLOON_FREE_PAGE_HINT_CMD_ID_MIN
        } else {
            self.free_page_hint_cmd_id + 1
        };
        self.free_page_hint_status = FreePageHintStatus::Requested;
        vdev.notify_config();
    }

    /// `virtio_balloon_free_page_stop()`: tells the guest to stop hinting.
    pub fn free_page_stop(&mut self, vdev: &mut VirtIODevice) {
        if self.free_page_hint_status != FreePageHintStatus::Stop {
            self.free_page_hint_status = FreePageHintStatus::Stop;
            vdev.notify_config();
        }
    }

    /// `virtio_balloon_free_page_done()`: tells the guest it may use the hinted pages again.
    pub fn free_page_done(&mut self, vdev: &mut VirtIODevice) {
        if self.free_page_hint_status != FreePageHintStatus::Done {
            self.free_page_hint_status = FreePageHintStatus::Done;
            vdev.notify_config();
        }
    }

    // Free page reporting.

    /// `virtio_balloon_handle_report()`.
    fn handle_report(&mut self, vdev: &mut VirtIODevice, vq: u16) {
        while let Some(elem) = vdev.pop(vq) {
            // Discarded pages come back zeroed, which is wrong if the guest poisons them.
            if !self.backend.discard_inhibited() && self.poison_val == 0 {
                for d in elem.writable() {
                    let (addr, len) = (d.addr(), u64::from(d.len()));
                    if !self.backend.is_ram(addr) {
                        continue;
                    }
                    let page_size = self.backend.host_page_size(addr).max(BALLOON_PAGE_SIZE);
                    // Unaligned ranges and ranges past the end of RAM are ignored.
                    if (addr | len) & (page_size - 1) != 0 {
                        continue;
                    }
                    if addr.checked_add(len).is_none_or(|end| end > self.ram_size) {
                        continue;
                    }
                    self.backend.discard(addr, len);
                }
            }
            vdev.push(vq, &elem, 0);
            vdev.notify(vq);
        }
    }

    fn config_size(&self) -> usize {
        if self.host_features & feature(VIRTIO_BALLOON_F_PAGE_POISON) != 0 {
            CONFIG_SIZE
        } else if self.host_features & feature(VIRTIO_BALLOON_F_FREE_PAGE_HINT) != 0 {
            CFG_POISON_VAL
        } else {
            CFG_FREE_PAGE_HINT_CMD_ID
        }
    }
}

impl VirtioDeviceClass for VirtioBalloon {
    fn realize(&mut self, vdev: &mut VirtIODevice) -> Result<()> {
        vdev.init(TYPE_VIRTIO_BALLOON, VIRTIO_ID_BALLOON, self.config_size());
        vdev.add_queue(BALLOON_QUEUE_SIZE)?;
        vdev.add_queue(BALLOON_QUEUE_SIZE)?;
        vdev.add_queue(BALLOON_QUEUE_SIZE)?;
        self.free_page_vq = None;
        self.reporting_vq = None;
        if self.host_features & feature(VIRTIO_BALLOON_F_FREE_PAGE_HINT) != 0 {
            self.free_page_vq = Some(vdev.add_queue(VIRTQUEUE_MAX_SIZE)?);
        }
        if self.host_features & feature(VIRTIO_BALLOON_F_REPORTING) != 0 {
            self.reporting_vq = Some(vdev.add_queue(REPORTING_QUEUE_SIZE)?);
        }
        self.stats = [u64::MAX; VIRTIO_BALLOON_S_NR];
        self.stats_last_update = 0;
        Ok(())
    }

    fn get_features(&mut self, _vdev: &VirtIODevice, features: u64) -> Result<u64> {
        Ok(features | self.host_features | feature(VIRTIO_BALLOON_F_STATS_VQ))
    }

    fn get_config(&mut self, _vdev: &VirtIODevice, config: &mut [u8]) {
        let mut c = [0u8; CONFIG_SIZE];
        c[CFG_NUM_PAGES..CFG_NUM_PAGES + 4].copy_from_slice(&self.num_pages.to_le_bytes());
        c[CFG_ACTUAL..CFG_ACTUAL + 4].copy_from_slice(&self.actual.to_le_bytes());
        let cmd_id = match self.free_page_hint_status {
            FreePageHintStatus::Requested => self.free_page_hint_cmd_id,
            FreePageHintStatus::Stop => VIRTIO_BALLOON_CMD_ID_STOP,
            FreePageHintStatus::Done => VIRTIO_BALLOON_CMD_ID_DONE,
            FreePageHintStatus::Start => 0,
        };
        c[CFG_FREE_PAGE_HINT_CMD_ID..CFG_FREE_PAGE_HINT_CMD_ID + 4]
            .copy_from_slice(&cmd_id.to_le_bytes());
        c[CFG_POISON_VAL..CFG_POISON_VAL + 4].copy_from_slice(&self.poison_val.to_le_bytes());
        let n = config.len().min(self.config_size());
        config[..n].copy_from_slice(&c[..n]);
    }

    fn set_config(&mut self, vdev: &mut VirtIODevice, config: &mut [u8]) {
        let word = |off: usize| -> Option<u32> {
            let b = config.get(off..off + 4)?;
            Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        };
        let old = self.actual;
        if let Some(actual) = word(CFG_ACTUAL) {
            self.actual = actual;
        }
        if self.actual != old {
            self.change_events.push(
                self.ram_size.wrapping_sub(u64::from(self.actual) << VIRTIO_BALLOON_PFN_SHIFT),
            );
        }
        self.poison_val = 0;
        if vdev.has_feature(VIRTIO_BALLOON_F_PAGE_POISON) {
            self.poison_val = word(CFG_POISON_VAL).unwrap_or(0);
        }
    }

    fn reset(&mut self, vdev: &mut VirtIODevice) {
        if vdev.has_feature(VIRTIO_BALLOON_F_FREE_PAGE_HINT) {
            self.free_page_stop(vdev);
        }
        if let Some(elem) = self.stats_elem.take() {
            vdev.detach(BALLOON_SVQ, &elem);
        }
        self.poison_val = 0;
    }

    fn handle_output(&mut self, vdev: &mut VirtIODevice, queue: u16) {
        match queue {
            BALLOON_IVQ | BALLOON_DVQ => self.handle_output_pages(vdev, queue),
            BALLOON_SVQ => self.receive_stats(vdev),
            q if Some(q) == self.free_page_vq => self.handle_free_page_vq(vdev, q),
            q if Some(q) == self.reporting_vq => self.handle_report(vdev, q),
            _ => {}
        }
    }

    fn post_load(&mut self, vdev: &mut VirtIODevice) -> Result<()> {
        self.vmstate_post_load(vdev)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
