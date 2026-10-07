// SPDX-License-Identifier: GPL-2.0-or-later

//! virtio-rng, a port of `hw/virtio/virtio-rng.c`.
//!
//! One queue of 8 entries. Whenever the guest makes buffers available and the device is ready,
//! the device asks its [`EntropySource`] for as many bytes as the buffers hold (bounded by the
//! rate limit quota) and fills them.
//!
//! Differences from QEMU:
//!
//! - QEMU's `RngBackend` is asynchronous: the device posts a request and the backend calls back
//!   later. Here [`EntropySource::fill`] answers on the spot and may return fewer bytes than
//!   asked, including none. Bytes that arrive some other way can be handed over with
//!   [`VirtioRng::entropy_available`], which is QEMU's `chr_read()`.
//! - The rate limit timer is not wired to a clock. The device records when QEMU would have armed
//!   it ([`VirtioRng::timer_armed`]) and whoever owns the device calls
//!   [`VirtioRng::check_rate_limit`] every [`VirtioRngConf::period_ms`] milliseconds. With the
//!   default `max-bytes` the quota never runs out, so nothing needs to call it.
//! - There is no VM run state, so the device always behaves as if the VM is running.
//!
//! Migration: the device has no state beyond the core's.
//!
//! Not ported: trace points, QOM registration and the `rng` link property.

use std::any::Any;
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use ruvm_base::{Error, Result};

use crate::virtio::{VIRTIO_CONFIG_S_DRIVER_OK, VirtIODevice, VirtioDeviceClass};

/// `TYPE_VIRTIO_RNG`.
pub const TYPE_VIRTIO_RNG: &str = "virtio-rng-device";

/// `VIRTIO_ID_RNG`.
pub const VIRTIO_ID_RNG: u16 = 4;

/// The queue size virtio-rng offers.
pub const VIRTIO_RNG_QUEUE_SIZE: u16 = 8;

/// Where entropy comes from, the synchronous counterpart of QEMU's `RngBackend`.
pub trait EntropySource: Send + fmt::Debug {
    /// Fills the front of `buf` with random bytes and returns how many were written. Returning
    /// fewer than `buf.len()`, or 0, is fine when there is not enough entropy right now.
    fn fill(&mut self, buf: &mut [u8]) -> usize;
}

/// Reads entropy from a file, by default `/dev/urandom`, like the `rng-random` backend.
#[derive(Debug)]
pub struct RandomFile {
    path: PathBuf,
    file: Option<File>,
}

impl RandomFile {
    /// Uses the file at `path`. It is opened on first use.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        RandomFile { path: path.into(), file: None }
    }
}

impl Default for RandomFile {
    /// `/dev/urandom`, the default `filename` of `rng-random`.
    fn default() -> Self {
        RandomFile::new("/dev/urandom")
    }
}

impl EntropySource for RandomFile {
    fn fill(&mut self, buf: &mut [u8]) -> usize {
        if self.file.is_none() {
            self.file = File::open(&self.path).ok();
        }
        match self.file.as_mut() {
            Some(f) => f.read(buf).unwrap_or(0),
            None => 0,
        }
    }
}

/// The virtio-rng properties.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VirtioRngConf {
    /// `max-bytes`: how many bytes the guest may get per period.
    pub max_bytes: u64,
    /// `period`: the length of a rate limit period in milliseconds.
    pub period_ms: u32,
}

impl Default for VirtioRngConf {
    fn default() -> Self {
        VirtioRngConf { max_bytes: i64::MAX as u64, period_ms: 1 << 16 }
    }
}

/// The virtio-rng device model, `VirtIORNG`.
#[derive(Debug)]
pub struct VirtioRng {
    conf: VirtioRngConf,
    source: Box<dyn EntropySource>,
    quota_remaining: i64,
    activate_timer: bool,
    timer_armed: bool,
}

impl VirtioRng {
    /// A device drawing from `source` with properties `conf`. The properties are checked when
    /// the device is realized.
    pub fn new(source: Box<dyn EntropySource>, conf: VirtioRngConf) -> Self {
        VirtioRng { conf, source, quota_remaining: 0, activate_timer: true, timer_armed: false }
    }

    /// The properties.
    pub fn conf(&self) -> VirtioRngConf {
        self.conf
    }

    /// How many more bytes the guest may get in this period.
    pub fn quota_remaining(&self) -> i64 {
        self.quota_remaining
    }

    /// Whether QEMU would have the rate limit timer running now. It is armed the first time the
    /// device has something to do after realize or after the previous expiry.
    pub fn timer_armed(&self) -> bool {
        self.timer_armed
    }

    /// `is_guest_ready()`.
    fn is_guest_ready(vdev: &VirtIODevice) -> bool {
        vdev.queue_ready(0) && vdev.status() & VIRTIO_CONFIG_S_DRIVER_OK != 0
    }

    /// `chr_read()`: hands `buf` to the guest, filling as many buffers as it takes. Returns how
    /// many bytes were used. Does nothing unless the driver is ready.
    pub fn entropy_available(&mut self, vdev: &mut VirtIODevice, buf: &[u8]) -> usize {
        if !Self::is_guest_ready(vdev) {
            return 0;
        }
        self.quota_remaining = self.quota_remaining.saturating_sub(buf.len() as i64);
        let mut offset = 0;
        while offset < buf.len() {
            let Some(chain) = vdev.pop(0) else {
                break;
            };
            let mem = std::sync::Arc::clone(vdev.mem());
            let mut w = chain.writer(&*mem);
            let len = w.write(&buf[offset..]).unwrap_or(0);
            offset += len;
            vdev.push(0, &chain, len as u32);
        }
        vdev.notify(0);
        offset
    }

    /// `virtio_rng_process()`: fills whatever the guest has made available, as far as the quota
    /// and the entropy source allow.
    ///
    /// QEMU requests more entropy from its completion callback while the queue is not empty.
    /// Here that is a loop, which stops as soon as the source has nothing to give.
    pub fn process(&mut self, vdev: &mut VirtIODevice) {
        loop {
            if !Self::is_guest_ready(vdev) {
                return;
            }
            if self.activate_timer {
                self.timer_armed = true;
                self.activate_timer = false;
            }
            let quota = if self.quota_remaining < 0 {
                0
            } else {
                (self.quota_remaining as u64).min(u64::from(u32::MAX))
            };
            let (size, _) = vdev.avail_bytes(0, quota, 0);
            if size == 0 {
                return;
            }
            let mut buf = vec![0; size as usize];
            let got = self.source.fill(&mut buf).min(buf.len());
            if got == 0 {
                return;
            }
            self.entropy_available(vdev, &buf[..got]);
            if vdev.queue_empty(0) {
                return;
            }
        }
    }

    /// `check_rate_limit()`: the rate limit timer expired. Refills the quota and serves any
    /// waiting buffers.
    pub fn check_rate_limit(&mut self, vdev: &mut VirtIODevice) {
        self.timer_armed = false;
        self.quota_remaining = self.conf.max_bytes.min(i64::MAX as u64) as i64;
        self.process(vdev);
        self.activate_timer = true;
    }
}

impl VirtioDeviceClass for VirtioRng {
    fn realize(&mut self, vdev: &mut VirtIODevice) -> Result<()> {
        if self.conf.period_ms == 0 {
            return Err(Error::generic("'period' parameter expects a positive integer"));
        }
        // Actually the max value is i64::MAX, a u64 property just makes the check simpler.
        if self.conf.max_bytes == 0 || self.conf.max_bytes > i64::MAX as u64 {
            return Err(Error::generic(
                "'max-bytes' parameter must be positive, and less than 2^63",
            ));
        }
        vdev.init(TYPE_VIRTIO_RNG, VIRTIO_ID_RNG, 0);
        vdev.add_queue(VIRTIO_RNG_QUEUE_SIZE)?;
        self.quota_remaining = self.conf.max_bytes as i64;
        self.activate_timer = true;
        self.timer_armed = false;
        Ok(())
    }

    fn set_status(&mut self, vdev: &mut VirtIODevice, status: u8) -> Result<()> {
        vdev.set_status_value(status);
        // Something changed, try to process buffers.
        self.process(vdev);
        Ok(())
    }

    fn handle_output(&mut self, vdev: &mut VirtIODevice, _queue: u16) {
        self.process(vdev);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
