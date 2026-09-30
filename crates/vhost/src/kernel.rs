// SPDX-License-Identifier: MIT OR Apache-2.0

//! The in-kernel vhost backends, `/dev/vhost-net` and `/dev/vhost-vsock`, driven through the
//! ioctls in the Linux UAPI header `linux/vhost.h`.
//!
//! The kernel backend runs in a kernel thread that shares the VMM's address space, so memory
//! regions and ring addresses are plain VMM virtual addresses and no file descriptors change
//! hands except eventfds and, for vhost-net, the tap device.

#![allow(unsafe_code)]

use std::ffi::{c_int, c_uint, c_ulong};
use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

use vmm_sys_util::ioctl::{ioctl, ioctl_with_mut_ref, ioctl_with_ptr, ioctl_with_ref};
use vmm_sys_util::{ioctl_io_nr, ioctl_ior_nr, ioctl_iow_nr, ioctl_iowr_nr};

use crate::{Error, LogRegion, MemoryRegion, Result, VhostBackend, VringAddr, check_regions};

/// The default path of the vhost-net device.
pub const VHOST_NET_PATH: &str = "/dev/vhost-net";
/// The default path of the vhost-vsock device.
pub const VHOST_VSOCK_PATH: &str = "/dev/vhost-vsock";

/// The most memory regions the kernel accepts by default (the `max_mem_regions` module
/// parameter of vhost).
pub const VHOST_KERNEL_DEFAULT_MAX_REGIONS: usize = 64;

/// `struct vhost_vring_state`.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
struct VringState {
    index: c_uint,
    num: c_uint,
}

/// `struct vhost_vring_file`. `fd` is -1 to unbind.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
struct VringFile {
    index: c_uint,
    fd: c_int,
}

/// `struct vhost_vring_addr`.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
struct VringAddrRaw {
    index: c_uint,
    flags: c_uint,
    desc_user_addr: u64,
    used_user_addr: u64,
    avail_user_addr: u64,
    log_guest_addr: u64,
}

/// `struct vhost_memory` without its flexible array of regions.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
struct MemoryHeader {
    nregions: u32,
    padding: u32,
}

/// `struct vhost_memory_region`.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
struct MemoryRegionRaw {
    guest_phys_addr: u64,
    memory_size: u64,
    userspace_addr: u64,
    flags_padding: u64,
}

const VHOST_VIRTIO: c_uint = 0xAF;

ioctl_ior_nr!(VHOST_GET_FEATURES, VHOST_VIRTIO, 0x00, u64);
ioctl_iow_nr!(VHOST_SET_FEATURES, VHOST_VIRTIO, 0x00, u64);
ioctl_io_nr!(VHOST_SET_OWNER, VHOST_VIRTIO, 0x01);
ioctl_io_nr!(VHOST_RESET_OWNER, VHOST_VIRTIO, 0x02);
ioctl_iow_nr!(VHOST_SET_MEM_TABLE, VHOST_VIRTIO, 0x03, MemoryHeader);
ioctl_iow_nr!(VHOST_SET_LOG_BASE, VHOST_VIRTIO, 0x04, u64);
ioctl_iow_nr!(VHOST_SET_LOG_FD, VHOST_VIRTIO, 0x07, c_int);
ioctl_iow_nr!(VHOST_SET_VRING_NUM, VHOST_VIRTIO, 0x10, VringState);
ioctl_iow_nr!(VHOST_SET_VRING_ADDR, VHOST_VIRTIO, 0x11, VringAddrRaw);
ioctl_iow_nr!(VHOST_SET_VRING_BASE, VHOST_VIRTIO, 0x12, VringState);
ioctl_iowr_nr!(VHOST_GET_VRING_BASE, VHOST_VIRTIO, 0x12, VringState);
ioctl_iow_nr!(VHOST_SET_VRING_KICK, VHOST_VIRTIO, 0x20, VringFile);
ioctl_iow_nr!(VHOST_SET_VRING_CALL, VHOST_VIRTIO, 0x21, VringFile);
ioctl_iow_nr!(VHOST_SET_VRING_ERR, VHOST_VIRTIO, 0x22, VringFile);
ioctl_iow_nr!(VHOST_NET_SET_BACKEND, VHOST_VIRTIO, 0x30, VringFile);
ioctl_iow_nr!(VHOST_VSOCK_SET_GUEST_CID, VHOST_VIRTIO, 0x60, u64);
ioctl_iow_nr!(VHOST_VSOCK_SET_RUNNING, VHOST_VIRTIO, 0x61, c_int);

/// Types that are exactly the argument some vhost ioctl reads or writes: `repr(C)` or a
/// primitive, with no padding the kernel could read uninitialized. The trait is private so the
/// set cannot grow outside this file.
trait IoctlArg: Copy {}
impl IoctlArg for u64 {}
impl IoctlArg for c_int {}
impl IoctlArg for VringState {}
impl IoctlArg for VringFile {}
impl IoctlArg for VringAddrRaw {}

fn check(ret: c_int) -> Result<()> {
    if ret < 0 { Err(Error::Io(io::Error::last_os_error())) } else { Ok(()) }
}

/// An open in-kernel vhost device.
#[derive(Debug)]
pub struct VhostKernel {
    file: File,
}

impl VhostKernel {
    /// Open the vhost device at `path` for reading and writing, non-blocking, as QEMU does.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC)
            .open(path)?;
        Ok(VhostKernel { file })
    }

    /// Open `/dev/vhost-net`.
    pub fn open_net() -> Result<Self> {
        VhostKernel::open(VHOST_NET_PATH)
    }

    /// Open `/dev/vhost-vsock`.
    pub fn open_vsock() -> Result<Self> {
        VhostKernel::open(VHOST_VSOCK_PATH)
    }

    /// Take over a vhost device file the VMM was handed, for `vhostfd=` on the command line.
    pub fn from_file(file: File) -> Self {
        VhostKernel { file }
    }

    /// The device file.
    pub fn file(&self) -> &File {
        &self.file
    }

    fn ioctl_none(&self, req: c_ulong) -> Result<()> {
        // SAFETY: `req` is one of the `_IO` requests above, which take no argument, so the
        // kernel reads and writes no memory of ours. The fd is open for the life of `self`.
        check(unsafe { ioctl(&self.file, req) })
    }

    /// Run an `_IOW` request that reads a `T` from us.
    fn ioctl_write<T: IoctlArg>(&self, req: c_ulong, arg: &T) -> Result<()> {
        // SAFETY: every caller pairs `req` with the `T` it was declared with in the
        // `ioctl_*_nr!` list above, so the kernel reads exactly `size_of::<T>()` bytes from a
        // live reference and writes nothing.
        check(unsafe { ioctl_with_ref(&self.file, req, arg) })
    }

    /// Run an `_IOR` or `_IOWR` request that writes a `T` back.
    fn ioctl_read_write<T: IoctlArg>(&self, req: c_ulong, arg: &mut T) -> Result<()> {
        // SAFETY: as for `ioctl_write`, `req` is declared with `T`, so the kernel reads and
        // writes at most `size_of::<T>()` bytes through a live exclusive reference, and any bit
        // pattern is a valid `T` for these plain integer structs.
        check(unsafe { ioctl_with_mut_ref(&self.file, req, arg) })
    }

    fn vring_file(&self, req: c_ulong, index: u32, fd: Option<BorrowedFd<'_>>) -> Result<()> {
        let fd = fd.map_or(-1, |fd| fd.as_raw_fd());
        self.ioctl_write(req, &VringFile { index, fd })
    }

    /// `VHOST_SET_OWNER`.
    pub fn set_owner(&mut self) -> Result<()> {
        self.ioctl_none(VHOST_SET_OWNER())
    }

    /// `VHOST_RESET_OWNER`.
    pub fn reset_owner(&mut self) -> Result<()> {
        self.ioctl_none(VHOST_RESET_OWNER())
    }

    /// `VHOST_GET_FEATURES`.
    pub fn get_features(&mut self) -> Result<u64> {
        let mut features = 0u64;
        self.ioctl_read_write(VHOST_GET_FEATURES(), &mut features)?;
        Ok(features)
    }

    /// `VHOST_SET_FEATURES`.
    pub fn set_features(&mut self, features: u64) -> Result<()> {
        self.ioctl_write(VHOST_SET_FEATURES(), &features)
    }

    /// `VHOST_SET_MEM_TABLE`. The file descriptors and offsets in `regions` are not used.
    pub fn set_mem_table(&mut self, regions: &[MemoryRegion<'_>]) -> Result<()> {
        if regions.len() > VHOST_KERNEL_DEFAULT_MAX_REGIONS {
            return Err(Error::TooManyRegions {
                count: regions.len(),
                max: VHOST_KERNEL_DEFAULT_MAX_REGIONS,
            });
        }
        check_regions(regions)?;
        // `struct vhost_memory` is a header followed by a flexible array of regions. Build it
        // in a buffer of u64s so it is aligned for both.
        let words_per_region = size_of::<MemoryRegionRaw>() / 8;
        let mut buf = vec![0u64; 1 + regions.len() * words_per_region];
        let header =
            MemoryHeader { nregions: u32::try_from(regions.len()).unwrap_or(u32::MAX), padding: 0 };
        let mut first = [0u8; 8];
        first[..4].copy_from_slice(&header.nregions.to_ne_bytes());
        first[4..].copy_from_slice(&header.padding.to_ne_bytes());
        buf[0] = u64::from_ne_bytes(first);
        for (i, r) in regions.iter().enumerate() {
            let at = 1 + i * words_per_region;
            buf[at] = r.guest_phys_addr;
            buf[at + 1] = r.memory_size;
            buf[at + 2] = r.userspace_addr;
            buf[at + 3] = 0;
        }
        // SAFETY: the buffer holds a `struct vhost_memory` header in its first word, `nregions`
        // in the low addressed four bytes, followed by exactly `nregions` `struct
        // vhost_memory_region`s, which is everything the kernel reads for this request. It
        // outlives the call and the kernel does not write to it.
        check(unsafe { ioctl_with_ptr(&self.file, VHOST_SET_MEM_TABLE(), buf.as_ptr()) })
    }

    /// `VHOST_SET_LOG_BASE`. `base` is the VMM address of the dirty log.
    pub fn set_log_base(&mut self, base: u64) -> Result<()> {
        self.ioctl_write(VHOST_SET_LOG_BASE(), &base)
    }

    /// `VHOST_SET_LOG_FD`.
    pub fn set_log_fd(&mut self, fd: BorrowedFd<'_>) -> Result<()> {
        self.ioctl_write(VHOST_SET_LOG_FD(), &fd.as_raw_fd())
    }

    /// `VHOST_SET_VRING_NUM`.
    pub fn set_vring_num(&mut self, index: u32, num: u32) -> Result<()> {
        self.ioctl_write(VHOST_SET_VRING_NUM(), &VringState { index, num })
    }

    /// `VHOST_SET_VRING_ADDR`.
    pub fn set_vring_addr(&mut self, addr: &VringAddr) -> Result<()> {
        let raw = VringAddrRaw {
            index: addr.index,
            flags: addr.flags,
            desc_user_addr: addr.desc_user_addr,
            used_user_addr: addr.used_user_addr,
            avail_user_addr: addr.avail_user_addr,
            log_guest_addr: addr.log_guest_addr,
        };
        self.ioctl_write(VHOST_SET_VRING_ADDR(), &raw)
    }

    /// `VHOST_SET_VRING_BASE`.
    pub fn set_vring_base(&mut self, index: u32, base: u32) -> Result<()> {
        self.ioctl_write(VHOST_SET_VRING_BASE(), &VringState { index, num: base })
    }

    /// `VHOST_GET_VRING_BASE`.
    pub fn get_vring_base(&mut self, index: u32) -> Result<u32> {
        let mut state = VringState { index, num: 0 };
        self.ioctl_read_write(VHOST_GET_VRING_BASE(), &mut state)?;
        Ok(state.num)
    }

    /// `VHOST_SET_VRING_KICK`.
    pub fn set_vring_kick(&mut self, index: u32, fd: Option<BorrowedFd<'_>>) -> Result<()> {
        self.vring_file(VHOST_SET_VRING_KICK(), index, fd)
    }

    /// `VHOST_SET_VRING_CALL`.
    pub fn set_vring_call(&mut self, index: u32, fd: Option<BorrowedFd<'_>>) -> Result<()> {
        self.vring_file(VHOST_SET_VRING_CALL(), index, fd)
    }

    /// `VHOST_SET_VRING_ERR`.
    pub fn set_vring_err(&mut self, index: u32, fd: Option<BorrowedFd<'_>>) -> Result<()> {
        self.vring_file(VHOST_SET_VRING_ERR(), index, fd)
    }

    /// `VHOST_NET_SET_BACKEND`: attach ring `index` to a tap device, or detach it with `None`.
    pub fn net_set_backend(&mut self, index: u32, tap: Option<BorrowedFd<'_>>) -> Result<()> {
        self.vring_file(VHOST_NET_SET_BACKEND(), index, tap)
    }

    /// `VHOST_VSOCK_SET_GUEST_CID`.
    pub fn vsock_set_guest_cid(&mut self, cid: u64) -> Result<()> {
        self.ioctl_write(VHOST_VSOCK_SET_GUEST_CID(), &cid)
    }

    /// `VHOST_VSOCK_SET_RUNNING`.
    pub fn vsock_set_running(&mut self, running: bool) -> Result<()> {
        self.ioctl_write(VHOST_VSOCK_SET_RUNNING(), &c_int::from(running))
    }
}

impl AsFd for VhostKernel {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.file.as_fd()
    }
}

impl VhostBackend for VhostKernel {
    fn set_owner(&mut self) -> Result<()> {
        VhostKernel::set_owner(self)
    }

    fn reset_owner(&mut self) -> Result<()> {
        VhostKernel::reset_owner(self)
    }

    fn get_features(&mut self) -> Result<u64> {
        VhostKernel::get_features(self)
    }

    fn set_features(&mut self, features: u64) -> Result<()> {
        VhostKernel::set_features(self, features)
    }

    fn set_mem_table(&mut self, regions: &[MemoryRegion<'_>]) -> Result<()> {
        VhostKernel::set_mem_table(self, regions)
    }

    fn set_log_base(&mut self, base: u64, _region: Option<LogRegion<'_>>) -> Result<()> {
        VhostKernel::set_log_base(self, base)
    }

    fn set_log_fd(&mut self, fd: BorrowedFd<'_>) -> Result<()> {
        VhostKernel::set_log_fd(self, fd)
    }

    fn set_vring_num(&mut self, index: u32, num: u32) -> Result<()> {
        VhostKernel::set_vring_num(self, index, num)
    }

    fn set_vring_addr(&mut self, addr: &VringAddr) -> Result<()> {
        VhostKernel::set_vring_addr(self, addr)
    }

    fn set_vring_base(&mut self, index: u32, base: u32) -> Result<()> {
        VhostKernel::set_vring_base(self, index, base)
    }

    fn get_vring_base(&mut self, index: u32) -> Result<u32> {
        VhostKernel::get_vring_base(self, index)
    }

    fn set_vring_kick(&mut self, index: u32, fd: Option<BorrowedFd<'_>>) -> Result<()> {
        VhostKernel::set_vring_kick(self, index, fd)
    }

    fn set_vring_call(&mut self, index: u32, fd: Option<BorrowedFd<'_>>) -> Result<()> {
        VhostKernel::set_vring_call(self, index, fd)
    }

    fn set_vring_err(&mut self, index: u32, fd: Option<BorrowedFd<'_>>) -> Result<()> {
        VhostKernel::set_vring_err(self, index, fd)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_numbers_match_the_uapi_header() {
        // Values from linux/vhost.h on x86-64 and aarch64, where _IOW puts the direction in
        // bits 30 and 31 and the size in bits 16 to 29.
        assert_eq!(VHOST_GET_FEATURES(), 0x8008_af00);
        assert_eq!(VHOST_SET_FEATURES(), 0x4008_af00);
        assert_eq!(VHOST_SET_OWNER(), 0x0000_af01);
        assert_eq!(VHOST_RESET_OWNER(), 0x0000_af02);
        assert_eq!(VHOST_SET_MEM_TABLE(), 0x4008_af03);
        assert_eq!(VHOST_SET_VRING_NUM(), 0x4008_af10);
        assert_eq!(VHOST_SET_VRING_ADDR(), 0x4028_af11);
        assert_eq!(VHOST_GET_VRING_BASE(), 0xc008_af12);
        assert_eq!(VHOST_SET_VRING_KICK(), 0x4008_af20);
        assert_eq!(VHOST_NET_SET_BACKEND(), 0x4008_af30);
        assert_eq!(VHOST_VSOCK_SET_GUEST_CID(), 0x4008_af60);
        assert_eq!(VHOST_VSOCK_SET_RUNNING(), 0x4004_af61);
        assert_eq!(size_of::<MemoryRegionRaw>(), 32);
    }
}
