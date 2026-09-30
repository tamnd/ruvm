// SPDX-License-Identifier: GPL-2.0-or-later

//! The vduse-blk export: block/export/vduse-blk.c with the part of subprojects/libvduse it uses.
//!
//! The export creates a virtio-blk device in the kernel through `/dev/vduse/control` and serves
//! its rings from a thread of its own. The thread waits on the device file for the kernel's
//! requests (status changes, IOTLB updates and ring state queries) and on one eventfd per ready
//! ring for the driver's kicks, and runs each request through
//! [`VirtioBlkHandler`](super::virtio_blk::VirtioBlkHandler). Guest buffers are reached through
//! the IOTLB files the kernel hands out, which are mapped as libvduse maps them.
//!
//! Differences from QEMU:
//!
//! - The reconnect log file `$TMPDIR/vduse-blk-<name>` is created with the size libvduse gives
//!   it, but in-flight descriptors are not recorded in it. A process that reattaches to an
//!   existing device therefore starts each ring at the used index and does not resubmit
//!   requests that were in flight when the previous process went away.
//! - A resize of the node is not passed on: the capacity in the config space is the one the
//!   device was created with.
//! - A request the handler drops as malformed does not keep counting as in flight, where QEMU
//!   leaks the count and so never lets the export go.
//! - A ring whose descriptors are broken prints the error of the ring code rather than the
//!   matching libvduse message, and stops being served until the next kick.

use std::cell::RefCell;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;

use rustix::event::{EventfdFlags, PollFd, PollFlags, eventfd, poll};
use rustix::io::Errno;
use ruvm_base::error::strerror;
use ruvm_base::{Error, Result, error_report};
use ruvm_qapi::types::BlockExportOptionsVduseBlk;
use ruvm_virtio_queue::{
    GuestMemory, MemoryError, RingAddresses, SplitQueue, VIRTIO_F_EVENT_IDX, VIRTIO_F_INDIRECT_DESC,
};

use super::virtio_blk::{
    ReqError, VIRTIO_BLK_F_BLK_SIZE, VIRTIO_BLK_F_DISCARD, VIRTIO_BLK_F_FLUSH, VIRTIO_BLK_F_MQ,
    VIRTIO_BLK_F_RO, VIRTIO_BLK_F_SEG_MAX, VIRTIO_BLK_F_TOPOLOGY, VIRTIO_BLK_F_WRITE_ZEROES,
    VIRTIO_BLK_MAX_DISCARD_SECTORS, VIRTIO_BLK_MAX_WRITE_ZEROES_SECTORS, VIRTIO_BLK_SECTOR_BITS,
    VIRTIO_BLK_SECTOR_SIZE, VIRTIO_F_IOMMU_PLATFORM, VIRTIO_F_NOTIFY_ON_EMPTY, VIRTIO_F_VERSION_1,
    VIRTIO_ID_BLOCK, VirtioBlkConfig, VirtioBlkHandler, check_block_size,
};
use super::{ExportArgs, ExportDriver};

/// `VDUSE_DEFAULT_NUM_QUEUE`
const VDUSE_DEFAULT_NUM_QUEUE: u16 = 1;
/// `VDUSE_DEFAULT_QUEUE_SIZE`
const VDUSE_DEFAULT_QUEUE_SIZE: u16 = 256;
/// `VIRTQUEUE_MAX_SIZE`
const VIRTQUEUE_MAX_SIZE: u32 = 1024;
/// `VDUSE_NAME_MAX`
const VDUSE_NAME_MAX: usize = 256;
/// `VDUSE_API_VERSION`
const VDUSE_API_VERSION: u64 = 0;
/// `VDUSE_VQ_ALIGN`
const VDUSE_VQ_ALIGN: u32 = 4096;
/// `VIRTIO_CONFIG_S_DRIVER_OK`
const VIRTIO_CONFIG_S_DRIVER_OK: u8 = 4;

/// `vduse_vq_log_size(VIRTQUEUE_MAX_SIZE)`: a 16 byte header and 16 bytes per descriptor,
/// rounded up to `LOG_ALIGNMENT`.
const VQ_LOG_SIZE: u64 = (16 * VIRTQUEUE_MAX_SIZE as u64 + 16).next_multiple_of(64);

/// The ioctl direction bits and size field of the host, from `asm/ioctl.h`.
#[cfg(any(
    target_arch = "powerpc",
    target_arch = "powerpc64",
    target_arch = "mips",
    target_arch = "mips64",
    target_arch = "sparc64"
))]
mod ioc {
    pub(super) const READ: u32 = 2;
    pub(super) const WRITE: u32 = 4;
    pub(super) const SIZEBITS: u32 = 13;
}
#[cfg(not(any(
    target_arch = "powerpc",
    target_arch = "powerpc64",
    target_arch = "mips",
    target_arch = "mips64",
    target_arch = "sparc64"
)))]
mod ioc {
    pub(super) const READ: u32 = 2;
    pub(super) const WRITE: u32 = 1;
    pub(super) const SIZEBITS: u32 = 14;
}

/// `_IOC(dir, VDUSE_BASE, nr, size)`
const fn vduse_ioc(dir: u32, nr: u32, size: usize) -> u32 {
    (dir << (16 + ioc::SIZEBITS)) | ((size as u32) << 16) | (0x81 << 8) | nr
}

/// The argument size encoded in an ioctl number.
const fn ioc_size(req: u32) -> usize {
    ((req >> 16) & ((1 << ioc::SIZEBITS) - 1)) as usize
}

/// `sizeof(struct vduse_dev_config)`, without the config space that follows it.
const DEV_CONFIG_SIZE: usize = 336;
/// `sizeof(struct vduse_iotlb_entry)`
const IOTLB_ENTRY_SIZE: usize = 32;
/// `sizeof(struct vduse_vq_config)`
const VQ_CONFIG_SIZE: usize = 32;
/// `sizeof(struct vduse_vq_info)`
const VQ_INFO_SIZE: usize = 48;
/// `sizeof(struct vduse_dev_request)` and `sizeof(struct vduse_dev_response)`
const DEV_MSG_SIZE: usize = 152;

const VDUSE_SET_API_VERSION: u32 = vduse_ioc(ioc::WRITE, 0x01, 8);
const VDUSE_CREATE_DEV: u32 = vduse_ioc(ioc::WRITE, 0x02, DEV_CONFIG_SIZE);
const VDUSE_DESTROY_DEV: u32 = vduse_ioc(ioc::WRITE, 0x03, VDUSE_NAME_MAX);
const VDUSE_IOTLB_GET_FD: u32 = vduse_ioc(ioc::READ | ioc::WRITE, 0x10, IOTLB_ENTRY_SIZE);
const VDUSE_DEV_GET_FEATURES: u32 = vduse_ioc(ioc::READ, 0x11, 8);
const VDUSE_VQ_SETUP: u32 = vduse_ioc(ioc::WRITE, 0x14, VQ_CONFIG_SIZE);
const VDUSE_VQ_GET_INFO: u32 = vduse_ioc(ioc::READ | ioc::WRITE, 0x15, VQ_INFO_SIZE);
const VDUSE_VQ_SETUP_KICKFD: u32 = vduse_ioc(ioc::WRITE, 0x16, 8);
const VDUSE_VQ_INJECT_IRQ: u32 = vduse_ioc(ioc::WRITE, 0x17, 4);

/// `VDUSE_EVENTFD_DEASSIGN`
const VDUSE_EVENTFD_DEASSIGN: i32 = -1;

/// `enum vduse_req_type`
const VDUSE_GET_VQ_STATE: u32 = 0;
const VDUSE_SET_STATUS: u32 = 1;
const VDUSE_UPDATE_IOTLB: u32 = 2;
/// `VDUSE_REQ_RESULT_*`
const VDUSE_REQ_RESULT_OK: u32 = 0;
const VDUSE_REQ_RESULT_FAILED: u32 = 1;
/// `VDUSE_ACCESS_*`
const VDUSE_ACCESS_RO: u8 = 1;
const VDUSE_ACCESS_WO: u8 = 2;

/// `ioctl(fd, req, arg)` for the VDUSE requests, which all take a pointer to a buffer at least
/// as large as the size encoded in `req`.
fn ioctl(fd: BorrowedFd<'_>, req: u32, arg: &mut [u8]) -> io::Result<i32> {
    assert!(arg.len() >= ioc_size(req));
    // SAFETY: every VDUSE request reads or writes at most the size encoded in its number, which
    // `arg` covers, except VDUSE_CREATE_DEV, which also reads `config_size` bytes of config space
    // after the header; create_dev() puts them in the same buffer. The buffer outlives the call
    // and nothing else refers to it meanwhile.
    let r = unsafe { libc::ioctl(fd.as_raw_fd(), req as _, arg.as_mut_ptr()) };
    if r < 0 { Err(io::Error::last_os_error()) } else { Ok(r) }
}

fn put32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_ne_bytes());
}

fn put64(b: &mut [u8], at: usize, v: u64) {
    b[at..at + 8].copy_from_slice(&v.to_ne_bytes());
}

/// A shared mapping of an IOTLB file.
#[derive(Debug)]
struct Mapping {
    addr: usize,
    len: usize,
}

impl Mapping {
    fn new(fd: BorrowedFd<'_>, len: usize, prot: libc::c_int) -> io::Result<Self> {
        // SAFETY: a new mapping at an address the kernel picks overlaps nothing Rust owns. The
        // file descriptor is valid for the call.
        let p = unsafe {
            libc::mmap(std::ptr::null_mut(), len, prot, libc::MAP_SHARED, fd.as_raw_fd(), 0)
        };
        if p == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(Mapping { addr: p as usize, len })
    }

    /// Copies out of the mapping at `off`, which the caller keeps within it.
    fn read(&self, off: usize, buf: &mut [u8]) {
        assert!(off.checked_add(buf.len()).is_some_and(|end| end <= self.len));
        // SAFETY: the range is inside a readable mapping that lives as long as `self`. The guest
        // may change the memory while it is copied, which gives the same torn bytes a DMA read
        // does but never touches memory Rust has references to.
        unsafe {
            std::ptr::copy_nonoverlapping(
                (self.addr + off) as *const u8,
                buf.as_mut_ptr(),
                buf.len(),
            );
        }
    }

    /// Copies into the mapping at `off`, which the caller keeps within it.
    fn write(&self, off: usize, buf: &[u8]) {
        assert!(off.checked_add(buf.len()).is_some_and(|end| end <= self.len));
        // SAFETY: the range is inside a writable mapping that lives as long as `self` and that
        // no Rust reference points into.
        unsafe {
            std::ptr::copy_nonoverlapping(buf.as_ptr(), (self.addr + off) as *mut u8, buf.len());
        }
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: the mapping was made by Mapping::new() and nothing refers into it any more.
        unsafe {
            libc::munmap(self.addr as *mut libc::c_void, self.len);
        }
    }
}

/// `VduseIovaRegion`
#[derive(Debug)]
struct IovaRegion {
    iova: u64,
    size: u64,
    /// Where `iova` is in the mapping.
    offset: u64,
    perm: u8,
    map: Mapping,
}

/// The device's view of guest memory: I/O virtual addresses, mapped from the kernel's IOTLB
/// files on first use as `iova_to_va()` does it.
#[derive(Debug)]
struct Iotlb {
    dev: File,
    regions: RefCell<Vec<IovaRegion>>,
}

impl Iotlb {
    /// `vduse_iova_add_region()` for the entry the kernel has for `iova`.
    fn fault(&self, iova: u64) -> bool {
        let mut entry = [0u8; IOTLB_ENTRY_SIZE];
        put64(&mut entry, 8, iova);
        put64(&mut entry, 16, iova + 1);
        let Ok(fd) = ioctl(self.dev.as_fd(), VDUSE_IOTLB_GET_FD, &mut entry) else {
            return false;
        };
        // SAFETY: VDUSE_IOTLB_GET_FD returns a new descriptor that nothing else owns.
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        let offset = u64::from_ne_bytes(entry[0..8].try_into().unwrap_or_default());
        let start = u64::from_ne_bytes(entry[8..16].try_into().unwrap_or_default());
        let last = u64::from_ne_bytes(entry[16..24].try_into().unwrap_or_default());
        let perm = entry[24];
        let prot = match perm {
            VDUSE_ACCESS_WO => libc::PROT_WRITE,
            VDUSE_ACCESS_RO => libc::PROT_READ,
            _ => libc::PROT_READ | libc::PROT_WRITE,
        };
        let size = last.wrapping_sub(start).wrapping_add(1);
        let Some(len) = size.checked_add(offset).and_then(|l| usize::try_from(l).ok()) else {
            return false;
        };
        let Ok(map) = Mapping::new(fd.as_fd(), len, prot) else {
            return false;
        };
        self.regions.borrow_mut().push(IovaRegion { iova: start, size, offset, perm, map });
        true
    }

    /// `vduse_iova_remove_region()`
    fn remove(&self, start: u64, last: u64) {
        if last == start {
            return;
        }
        self.regions.borrow_mut().retain(|r| !(start <= r.iova && last >= r.iova + (r.size - 1)));
    }

    /// Runs `f` on each piece of `addr..addr + len` with its region and the offset in the
    /// region's mapping.
    fn each(
        &self,
        addr: u64,
        len: usize,
        write: bool,
        mut f: impl FnMut(&Mapping, usize, std::ops::Range<usize>),
    ) -> std::result::Result<(), MemoryError> {
        let fail = || MemoryError::OutOfRange { addr, len: len as u64 };
        let mut done = 0usize;
        let mut faulted = false;
        while done < len {
            let at = addr.checked_add(done as u64).ok_or_else(fail)?;
            let regions = self.regions.borrow();
            let Some(r) = regions.iter().find(|r| at >= r.iova && at - r.iova < r.size) else {
                drop(regions);
                if faulted || !self.fault(at) {
                    return Err(fail());
                }
                faulted = true;
                continue;
            };
            faulted = false;
            let denied = if write { VDUSE_ACCESS_RO } else { VDUSE_ACCESS_WO };
            if r.perm == denied {
                return Err(fail());
            }
            let within = at - r.iova;
            let n = ((r.size - within).min((len - done) as u64)) as usize;
            let off = usize::try_from(r.offset + within).map_err(|_| fail())?;
            if off.checked_add(n).is_none_or(|end| end > r.map.len) {
                return Err(fail());
            }
            f(&r.map, off, done..done + n);
            done += n;
        }
        Ok(())
    }

    /// Whether `len` bytes at `iova` can be reached, `iova_to_va()` with a length check.
    fn reachable(&self, iova: u64, len: usize) -> bool {
        self.each(iova, len, false, |_, _, _| {}).is_ok()
    }
}

impl GuestMemory for Iotlb {
    fn read(&self, addr: u64, buf: &mut [u8]) -> std::result::Result<(), MemoryError> {
        let len = buf.len();
        self.each(addr, len, false, |m, off, range| m.read(off, &mut buf[range]))
    }

    fn write(&self, addr: u64, buf: &[u8]) -> std::result::Result<(), MemoryError> {
        self.each(addr, buf.len(), true, |m, off, range| m.write(off, &buf[range]))
    }
}

/// `VduseVirtq`
#[derive(Debug, Default)]
struct Virtq {
    index: u32,
    queue: Option<SplitQueue>,
    /// The kick eventfd while the ring is ready.
    kick: Option<OwnedFd>,
    last_avail: u16,
}

/// What the export handle and the device thread share.
#[derive(Debug)]
struct Shared {
    handler: VirtioBlkHandler,
    inflight: AtomicU32,
    stop: OwnedFd,
}

/// `VduseDev` with the rings, owned by the device thread.
#[derive(Debug)]
struct Device {
    iotlb: Iotlb,
    features: u64,
    vqs: Vec<Virtq>,
}

/// A running vduse-blk export.
#[derive(Debug)]
pub struct VduseBlkExport {
    shared: Arc<Shared>,
    thread: Mutex<Option<JoinHandle<Device>>>,
    ctrl: File,
    name: String,
    recon_file: String,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl ExportDriver for VduseBlkExport {
    /// The export is held while requests are in flight, `vduse_blk_inflight_inc()`.
    fn in_use(&self) -> bool {
        self.shared.inflight.load(Ordering::SeqCst) > 0
    }

    /// `vduse_blk_exp_request_shutdown()` and then `vduse_blk_exp_delete()`.
    fn shutdown(&self) {
        let Some(thread) = lock(&self.thread).take() else {
            return;
        };
        let _ = rustix::io::write(&self.shared.stop, &1u64.to_ne_bytes());
        let dev = thread.join();
        drop(dev);
        if destroy_dev(&self.ctrl, &self.name).map_err(|e| e.raw_os_error())
            != Err(Some(libc::EBUSY))
        {
            let _ = std::fs::remove_file(&self.recon_file);
        }
    }
}

impl Drop for VduseBlkExport {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// `VDUSE_DESTROY_DEV`
fn destroy_dev(ctrl: &File, name: &str) -> io::Result<()> {
    let mut buf = [0u8; VDUSE_NAME_MAX];
    let n = name.len().min(VDUSE_NAME_MAX - 1);
    buf[..n].copy_from_slice(&name.as_bytes()[..n]);
    ioctl(ctrl.as_fd(), VDUSE_DESTROY_DEV, &mut buf).map(|_| ())
}

/// `vduse_get_virtio_features()`
fn vduse_virtio_features() -> u64 {
    1 << VIRTIO_F_IOMMU_PLATFORM
        | 1 << VIRTIO_F_VERSION_1
        | 1 << VIRTIO_F_NOTIFY_ON_EMPTY
        | 1 << VIRTIO_F_EVENT_IDX
        | 1 << VIRTIO_F_INDIRECT_DESC
}

/// The features `vduse_blk_exp_create()` gives the device.
fn device_features(num_queues: u16, writable: bool) -> u64 {
    let mut features = vduse_virtio_features()
        | 1 << VIRTIO_BLK_F_SEG_MAX
        | 1 << VIRTIO_BLK_F_TOPOLOGY
        | 1 << VIRTIO_BLK_F_BLK_SIZE
        | 1 << VIRTIO_BLK_F_FLUSH
        | 1 << VIRTIO_BLK_F_DISCARD
        | 1 << VIRTIO_BLK_F_WRITE_ZEROES;
    if num_queues > 1 {
        features |= 1 << VIRTIO_BLK_F_MQ;
    }
    if !writable {
        features |= 1 << VIRTIO_BLK_F_RO;
    }
    features
}

/// `vduse_name_is_invalid()`
fn name_is_invalid(name: &str) -> bool {
    name.len() >= VDUSE_NAME_MAX || name.contains("..") || name.contains('\0')
}

/// `vduse_blk_exp_create()`.
pub fn create(
    args: &ExportArgs<'_>,
    opts: &BlockExportOptionsVduseBlk,
) -> Result<Arc<dyn ExportDriver>> {
    let num_queues = opts.num_queues.unwrap_or(VDUSE_DEFAULT_NUM_QUEUE);
    if num_queues == 0 {
        return Err(Error::generic("num-queues must be greater than 0"));
    }
    let queue_size = opts.queue_size.unwrap_or(VDUSE_DEFAULT_QUEUE_SIZE);
    if queue_size <= 2
        || !queue_size.is_power_of_two()
        || u32::from(queue_size) > VIRTQUEUE_MAX_SIZE
    {
        return Err(Error::generic("queue-size is invalid"));
    }
    let logical_block_size = opts.logical_block_size.unwrap_or(VIRTIO_BLK_SECTOR_SIZE);
    check_block_size("logical-block-size", logical_block_size)?;
    // check_block_size() keeps it at 2 MiB or below.
    let logical_block_size = logical_block_size as u32;

    let handler = VirtioBlkHandler {
        blk: args.blk.clone(),
        serial: opts.serial.clone().unwrap_or_default(),
        logical_block_size,
        writable: args.writable,
    };
    let config = VirtioBlkConfig {
        capacity: handler.blk.getlength().unwrap_or(0) >> VIRTIO_BLK_SECTOR_BITS,
        size_max: 0,
        seg_max: u32::from(queue_size) - 2,
        blk_size: logical_block_size,
        min_io_size: 1,
        opt_io_size: 1,
        wce: 0,
        num_queues,
        max_discard_sectors: VIRTIO_BLK_MAX_DISCARD_SECTORS,
        max_discard_seg: 1,
        discard_sector_alignment: logical_block_size >> VIRTIO_BLK_SECTOR_BITS,
        max_write_zeroes_sectors: VIRTIO_BLK_MAX_WRITE_ZEROES_SECTORS,
        max_write_zeroes_seg: 1,
    };
    let features = device_features(num_queues, args.writable);

    let Some((ctrl, dev)) = create_dev(&opts.name, features, num_queues, &config.to_bytes()) else {
        return Err(Error::generic("failed to create vduse device"));
    };
    let fail = |ctrl: &File| {
        let _ = destroy_dev(ctrl, &opts.name);
    };

    let recon_file = format!(
        "{}/vduse-blk-{}",
        std::env::temp_dir().to_string_lossy().trim_end_matches('/'),
        opts.name
    );
    if set_reconnect_log_file(&recon_file, num_queues).is_err() {
        eprintln!("Failed to get vduse log");
        drop(dev);
        fail(&ctrl);
        return Err(Error::generic("failed to set reconnect log file"));
    }

    let stop = match eventfd(0, EventfdFlags::CLOEXEC | EventfdFlags::NONBLOCK) {
        Ok(fd) => fd,
        Err(e) => {
            drop(dev);
            fail(&ctrl);
            return Err(Error::from_io("Failed to create eventfd", e.into()));
        }
    };
    let mut device = Device {
        iotlb: Iotlb { dev, regions: RefCell::new(Vec::new()) },
        features: 0,
        vqs: (0..u32::from(num_queues))
            .map(|index| Virtq { index, ..Default::default() })
            .collect(),
    };
    let shared = Arc::new(Shared { handler, inflight: AtomicU32::new(0), stop });
    for i in 0..usize::from(num_queues) {
        device.setup_queue(i, queue_size);
    }

    let sh = shared.clone();
    let thread =
        match std::thread::Builder::new().name(format!("vduse-blk {}", args.id)).spawn(move || {
            device.run(&sh);
            device
        }) {
            Ok(t) => t,
            Err(e) => {
                fail(&ctrl);
                let _ = std::fs::remove_file(&recon_file);
                return Err(Error::from_io("Failed to create thread", e));
            }
        };
    Ok(Arc::new(VduseBlkExport {
        shared,
        thread: Mutex::new(Some(thread)),
        ctrl,
        name: opts.name.clone(),
        recon_file,
    }))
}

/// `vduse_dev_create()`: the control file and the device file, with libvduse's messages.
fn create_dev(name: &str, features: u64, num_queues: u16, config: &[u8]) -> Option<(File, File)> {
    if name_is_invalid(name) || features & (1 << VIRTIO_F_VERSION_1) == 0 {
        eprintln!("Invalid parameter for vduse");
        return None;
    }
    let ctrl = match OpenOptions::new().read(true).write(true).open("/dev/vduse/control") {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to open /dev/vduse/control: {}", strerror(&e));
            return None;
        }
    };
    let mut version = VDUSE_API_VERSION.to_ne_bytes();
    if let Err(e) = ioctl(ctrl.as_fd(), VDUSE_SET_API_VERSION, &mut version) {
        eprintln!("Failed to set api version {VDUSE_API_VERSION}: {}", strerror(&e));
        return None;
    }

    let mut dev_config = vec![0u8; DEV_CONFIG_SIZE + config.len()];
    dev_config[..name.len()].copy_from_slice(name.as_bytes());
    put32(&mut dev_config, 256, 0);
    put32(&mut dev_config, 260, VIRTIO_ID_BLOCK);
    put64(&mut dev_config, 264, features);
    put32(&mut dev_config, 272, u32::from(num_queues));
    put32(&mut dev_config, 276, VDUSE_VQ_ALIGN);
    put32(&mut dev_config, DEV_CONFIG_SIZE - 4, config.len() as u32);
    dev_config[DEV_CONFIG_SIZE..].copy_from_slice(config);
    if let Err(e) = ioctl(ctrl.as_fd(), VDUSE_CREATE_DEV, &mut dev_config) {
        if e.raw_os_error() != Some(libc::EEXIST) {
            eprintln!("Failed to create vduse device {name}: {}", strerror(&e));
            return None;
        }
    }

    // vduse_dev_init()
    let dev = match OpenOptions::new().read(true).write(true).open(format!("/dev/vduse/{name}")) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to open vduse dev {name}: {}", strerror(&e));
            eprintln!("Failed to init vduse device {name}: {}", strerror(&e));
            let _ = destroy_dev(&ctrl, name);
            return None;
        }
    };
    let mut f = [0u8; 8];
    if let Err(e) = ioctl(dev.as_fd(), VDUSE_DEV_GET_FEATURES, &mut f) {
        eprintln!("Failed to get features: {}", strerror(&e));
        eprintln!("Failed to init vduse device {name}: {}", strerror(&e));
        drop(dev);
        let _ = destroy_dev(&ctrl, name);
        return None;
    }
    Some((ctrl, dev))
}

/// `vduse_log_get()` for `vduse_set_reconnect_log_file()`: the file, at its full size.
fn set_reconnect_log_file(path: &str, num_queues: u16) -> io::Result<()> {
    use std::os::unix::fs::OpenOptionsExt;
    let f = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .open(path)?;
    f.set_len(u64::from(num_queues) * VQ_LOG_SIZE)
}

impl Device {
    fn fd(&self) -> BorrowedFd<'_> {
        self.iotlb.dev.as_fd()
    }

    /// `vduse_dev_setup_queue()`
    fn setup_queue(&mut self, i: usize, max_size: u16) {
        let mut cfg = [0u8; VQ_CONFIG_SIZE];
        put32(&mut cfg, 0, self.vqs[i].index);
        cfg[4..6].copy_from_slice(&max_size.to_ne_bytes());
        if ioctl(self.fd(), VDUSE_VQ_SETUP, &mut cfg).is_err() {
            return;
        }
        self.enable_queue(i);
    }

    /// `vduse_queue_enable()` followed by `vduse_blk_enable_queue()`.
    fn enable_queue(&mut self, i: usize) {
        let index = self.vqs[i].index;
        let mut info = [0u8; VQ_INFO_SIZE];
        put32(&mut info, 0, index);
        if let Err(e) = ioctl(self.fd(), VDUSE_VQ_GET_INFO, &mut info) {
            eprintln!("Failed to get vq[{index}] info: {}", strerror(&e));
            return;
        }
        let ne32 = |at: usize| u32::from_ne_bytes(info[at..at + 4].try_into().unwrap_or_default());
        let ne64 = |at: usize| u64::from_ne_bytes(info[at..at + 8].try_into().unwrap_or_default());
        if info[40] == 0 {
            return;
        }
        let num = ne32(4);
        if num > VIRTQUEUE_MAX_SIZE {
            eprintln!("vq[{index}] vring num {num} exceeds max {VIRTQUEUE_MAX_SIZE}");
            return;
        }
        let addrs =
            RingAddresses { desc_table: ne64(8), driver_area: ne64(16), device_area: ne64(24) };
        if !self.update_vring(index, addrs) {
            eprintln!("Failed to update vring for vq[{index}]");
            return;
        }
        let Ok(kick) = eventfd(0, EventfdFlags::NONBLOCK | EventfdFlags::CLOEXEC) else {
            eprintln!("Failed to init eventfd for vq[{index}]");
            return;
        };
        let mut efd = [0u8; 8];
        put32(&mut efd, 0, index);
        efd[4..8].copy_from_slice(&kick.as_raw_fd().to_ne_bytes());
        if ioctl(self.fd(), VDUSE_VQ_SETUP_KICKFD, &mut efd).is_err() {
            eprintln!("Failed to setup kick fd for vq[{index}]");
            return;
        }
        let Ok(mut queue) = SplitQueue::new(num as u16, addrs) else {
            eprintln!("Failed to update vring for vq[{index}]");
            return;
        };
        queue.set_event_idx(self.features & (1 << VIRTIO_F_EVENT_IDX) != 0);
        queue.set_indirect_desc(self.features & (1 << VIRTIO_F_INDIRECT_DESC) != 0);

        // vduse_queue_check_inflights(), with nothing ever logged in flight.
        let Ok(used) = self.iotlb.read_u16(addrs.device_area + 2) else {
            eprintln!("Failed to check inflights for vq[{index}]");
            return;
        };
        queue.set_next_used(used);
        queue.set_next_avail(used);
        self.vqs[i].last_avail = used;
        let _ = self.inject_irq(index);

        // Make sure we don't miss any kick after reconnecting
        let _ = rustix::io::write(&kick, &1u64.to_ne_bytes());
        self.vqs[i].queue = Some(queue);
        self.vqs[i].kick = Some(kick);
    }

    /// `vduse_queue_update_vring()`
    fn update_vring(&self, index: u32, addrs: RingAddresses) -> bool {
        let ok = self.iotlb.reachable(addrs.desc_table, 16)
            && self.iotlb.reachable(addrs.driver_area, 4)
            && self.iotlb.reachable(addrs.device_area, 4);
        if !ok {
            eprintln!("Failed to get vq[{index}] iova mapping");
        }
        ok
    }

    /// `vduse_queue_disable()`
    fn disable_queue(&mut self, i: usize) {
        let Some(kick) = self.vqs[i].kick.take() else {
            return;
        };
        let mut efd = [0u8; 8];
        put32(&mut efd, 0, self.vqs[i].index);
        efd[4..8].copy_from_slice(&VDUSE_EVENTFD_DEASSIGN.to_ne_bytes());
        let _ = ioctl(self.fd(), VDUSE_VQ_SETUP_KICKFD, &mut efd);
        drop(kick);
        if let Some(q) = self.vqs[i].queue.take() {
            self.vqs[i].last_avail = q.next_avail();
        }
    }

    fn inject_irq(&self, index: u32) -> io::Result<()> {
        let mut b = index.to_ne_bytes();
        ioctl(self.fd(), VDUSE_VQ_INJECT_IRQ, &mut b).map(|_| ())
    }

    /// `vduse_dev_start_dataplane()`
    fn start_dataplane(&mut self) {
        let mut f = [0u8; 8];
        if let Err(e) = ioctl(self.fd(), VDUSE_DEV_GET_FEATURES, &mut f) {
            eprintln!("Failed to get features: {}", strerror(&e));
            return;
        }
        self.features = u64::from_ne_bytes(f);
        for i in 0..self.vqs.len() {
            self.enable_queue(i);
        }
    }

    /// `vduse_dev_stop_dataplane()`
    fn stop_dataplane(&mut self) {
        for i in 0..self.vqs.len() {
            self.disable_queue(i);
        }
        self.features = 0;
        self.iotlb.remove(0, u64::MAX);
    }

    /// `vduse_dev_handler()`: one request from the kernel. False when the device file failed.
    fn handle_request(&mut self) -> bool {
        let mut req = [0u8; DEV_MSG_SIZE];
        match rustix::io::read(self.fd(), &mut req) {
            Ok(DEV_MSG_SIZE) => {}
            Err(Errno::AGAIN) | Err(Errno::INTR) => return true,
            Ok(n) => {
                eprintln!("Read request error [{n}]: Success");
                return false;
            }
            Err(e) => {
                eprintln!("Read request error [-1]: {}", strerror(&e.into()));
                return false;
            }
        }
        let ty = u32::from_ne_bytes(req[0..4].try_into().unwrap_or_default());
        let mut resp = [0u8; DEV_MSG_SIZE];
        resp[0..4].copy_from_slice(&req[4..8]);
        let ne32 = |at: usize| u32::from_ne_bytes(req[at..at + 4].try_into().unwrap_or_default());
        let ne64 = |at: usize| u64::from_ne_bytes(req[at..at + 8].try_into().unwrap_or_default());
        let result = match ty {
            VDUSE_GET_VQ_STATE => match self.vqs.get(ne32(24) as usize) {
                Some(vq) => {
                    let avail = vq.queue.as_ref().map_or(vq.last_avail, SplitQueue::next_avail);
                    resp[28..30].copy_from_slice(&avail.to_ne_bytes());
                    VDUSE_REQ_RESULT_OK
                }
                None => VDUSE_REQ_RESULT_FAILED,
            },
            VDUSE_SET_STATUS => {
                let status = req[24];
                if status & VIRTIO_CONFIG_S_DRIVER_OK != 0 {
                    self.start_dataplane();
                } else if status == 0 {
                    self.stop_dataplane();
                }
                VDUSE_REQ_RESULT_OK
            }
            VDUSE_UPDATE_IOTLB => {
                // The iova will be updated by iova_to_va() later, so just remove it
                self.iotlb.remove(ne64(24), ne64(32));
                for vq in &self.vqs {
                    if let Some(q) = &vq.queue {
                        if !self.update_vring(vq.index, q.addresses()) {
                            eprintln!("Failed to update vring for vq[{}]", vq.index);
                        }
                    }
                }
                VDUSE_REQ_RESULT_OK
            }
            _ => VDUSE_REQ_RESULT_FAILED,
        };
        resp[4..8].copy_from_slice(&result.to_ne_bytes());
        match rustix::io::write(self.fd(), &resp) {
            Ok(DEV_MSG_SIZE) => true,
            Ok(n) => {
                eprintln!("Write request {ty} error [{n}]: Success");
                false
            }
            Err(e) => {
                eprintln!("Write request {ty} error [-1]: {}", strerror(&e.into()));
                false
            }
        }
    }

    /// `vduse_blk_vq_handler()`: every available request on ring `i`.
    fn process(&mut self, i: usize, sh: &Shared) {
        let on_empty = self.features & (1 << VIRTIO_F_NOTIFY_ON_EMPTY) != 0;
        let index = self.vqs[i].index;
        let Some(queue) = self.vqs[i].queue.as_mut() else {
            return;
        };
        let mem = &self.iotlb;
        loop {
            let chain = match queue.pop(mem) {
                Ok(Some(c)) => c,
                Ok(None) => return,
                Err(e) => {
                    eprintln!("{e}");
                    return;
                }
            };
            sh.inflight.fetch_add(1, Ordering::SeqCst);
            let r = sh.handler.process_req(mem, &chain);
            let in_len = match r {
                Ok(n) => n,
                Err(ReqError::Malformed) => {
                    sh.inflight.fetch_sub(1, Ordering::SeqCst);
                    continue;
                }
                Err(ReqError::Memory(_)) => {
                    sh.inflight.fetch_sub(1, Ordering::SeqCst);
                    eprintln!("virtio: invalid address for buffers");
                    return;
                }
            };
            // vduse_blk_req_complete(): vduse_queue_push() and vduse_queue_notify().
            let notify = queue.add_used(mem, chain.head(), in_len).and_then(|()| {
                let n = queue.needs_notification(mem)?;
                Ok(n || (on_empty && !queue.has_available(mem)?))
            });
            sh.inflight.fetch_sub(1, Ordering::SeqCst);
            match notify {
                Ok(true) => {
                    let mut b = index.to_ne_bytes();
                    if let Err(e) = ioctl(mem.dev.as_fd(), VDUSE_VQ_INJECT_IRQ, &mut b) {
                        eprintln!("Error inject irq for vq {index}: {}", strerror(&e));
                    }
                }
                Ok(false) => {}
                Err(e) => {
                    eprintln!("{e}");
                    return;
                }
            }
        }
    }

    /// The device thread: the kernel's requests and the rings' kicks until the export stops.
    fn run(&mut self, sh: &Shared) {
        loop {
            let ready: Vec<usize> =
                (0..self.vqs.len()).filter(|&i| self.vqs[i].kick.is_some()).collect();
            let (stop, dev, kicks) = {
                let mut fds = vec![
                    PollFd::new(&sh.stop, PollFlags::IN),
                    PollFd::new(&self.iotlb.dev, PollFlags::IN),
                ];
                for &i in &ready {
                    if let Some(k) = &self.vqs[i].kick {
                        fds.push(PollFd::new(k, PollFlags::IN));
                    }
                }
                match poll(&mut fds, None) {
                    Ok(_) => {}
                    Err(Errno::INTR) => continue,
                    Err(_) => break,
                }
                let kicks: Vec<bool> = fds[2..].iter().map(|f| !f.revents().is_empty()).collect();
                (!fds[0].revents().is_empty(), !fds[1].revents().is_empty(), kicks)
            };
            if stop {
                break;
            }
            if dev && !self.handle_request() {
                break;
            }
            for (&i, kicked) in ready.iter().zip(kicks) {
                if !kicked {
                    continue;
                }
                // on_vduse_vq_kick(); the ring may have gone with a request just handled.
                let Some(kick) = &self.vqs[i].kick else {
                    continue;
                };
                let mut b = [0u8; 8];
                if rustix::io::read(kick, &mut b).is_err() {
                    error_report("failed to read data from eventfd");
                    continue;
                }
                self.process(i, sh);
            }
        }
        // vduse_blk_stop_virtqueues()
        for i in 0..self.vqs.len() {
            self.disable_queue(i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ioctl_numbers() {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            assert_eq!(VDUSE_SET_API_VERSION, 0x4008_8101);
            assert_eq!(VDUSE_CREATE_DEV, 0x4150_8102);
            assert_eq!(VDUSE_DESTROY_DEV, 0x4100_8103);
            assert_eq!(VDUSE_IOTLB_GET_FD, 0xc020_8110);
            assert_eq!(VDUSE_DEV_GET_FEATURES, 0x8008_8111);
            assert_eq!(VDUSE_VQ_SETUP, 0x4020_8114);
            assert_eq!(VDUSE_VQ_GET_INFO, 0xc030_8115);
            assert_eq!(VDUSE_VQ_SETUP_KICKFD, 0x4008_8116);
            assert_eq!(VDUSE_VQ_INJECT_IRQ, 0x4004_8117);
        }
        assert_eq!(ioc_size(VDUSE_CREATE_DEV), DEV_CONFIG_SIZE);
        assert_eq!(VQ_LOG_SIZE, 16448);
    }

    #[test]
    fn features() {
        let f = device_features(1, true);
        assert_ne!(f & (1 << VIRTIO_F_IOMMU_PLATFORM), 0);
        assert_eq!(f & (1 << VIRTIO_BLK_F_MQ), 0);
        assert_eq!(f & (1 << VIRTIO_BLK_F_RO), 0);
        let f = device_features(2, false);
        assert_ne!(f & (1 << VIRTIO_BLK_F_MQ), 0);
        assert_ne!(f & (1 << VIRTIO_BLK_F_RO), 0);
    }

    #[test]
    fn names() {
        assert!(!name_is_invalid("vduse0"));
        assert!(name_is_invalid("a/../b"));
        assert!(name_is_invalid(&"x".repeat(256)));
    }
}
