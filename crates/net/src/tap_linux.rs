// SPDX-License-Identifier: GPL-2.0-or-later

//! The Linux side of the tap backend, net/tap-linux.c.
//!
//! Every ioctl goes through [`ioctl_ref`] or [`ioctl_val`], which are the only unsafe code here.

#![allow(unsafe_code)]

use std::fs::OpenOptions;
use std::io;
use std::os::fd::{AsRawFd, OwnedFd};
use std::os::raw::c_ulong;

use ruvm_base::error::strerror;
use ruvm_base::{Error, Result, error_report, warn_report};

use crate::client::NetOffloads;

const PATH_NET_TUN: &str = "/dev/net/tun";

const IFNAMSIZ: usize = 16;

const IFF_TAP: i16 = 0x0002;
const IFF_NO_PI: i16 = 0x1000;
const IFF_ONE_QUEUE: u32 = 0x2000;
const IFF_VNET_HDR: u32 = 0x4000;
const IFF_MULTI_QUEUE: u32 = 0x0100;
const IFF_ATTACH_QUEUE: i16 = 0x0200;
const IFF_DETACH_QUEUE: i16 = 0x0400;

const TUN_F_CSUM: u32 = 0x01;
const TUN_F_TSO4: u32 = 0x02;
const TUN_F_TSO6: u32 = 0x04;
const TUN_F_TSO_ECN: u32 = 0x08;
const TUN_F_UFO: u32 = 0x10;
const TUN_F_USO4: u32 = 0x20;
const TUN_F_USO6: u32 = 0x40;
const TUN_F_UDP_TUNNEL_GSO: u32 = 0x080;
const TUN_F_UDP_TUNNEL_GSO_CSUM: u32 = 0x100;

/// The ioctl numbers, from linux/if_tun.h. The macro makes them public functions, so they live
/// in a module of their own where that is fine.
#[allow(unreachable_pub)]
mod nr {
    use std::os::raw::{c_int, c_uint};

    const TUNTAP: c_uint = b'T' as c_uint;

    vmm_sys_util::ioctl_iow_nr!(TUNSETIFF, TUNTAP, 202, c_int);
    vmm_sys_util::ioctl_ior_nr!(TUNGETFEATURES, TUNTAP, 207, c_uint);
    vmm_sys_util::ioctl_iow_nr!(TUNSETOFFLOAD, TUNTAP, 208, c_uint);
    vmm_sys_util::ioctl_ior_nr!(TUNGETIFF, TUNTAP, 210, c_uint);
    vmm_sys_util::ioctl_iow_nr!(TUNSETSNDBUF, TUNTAP, 212, c_int);
    vmm_sys_util::ioctl_iow_nr!(TUNSETVNETHDRSZ, TUNTAP, 216, c_int);
    vmm_sys_util::ioctl_iow_nr!(TUNSETQUEUE, TUNTAP, 217, c_int);
    vmm_sys_util::ioctl_iow_nr!(TUNSETVNETLE, TUNTAP, 220, c_int);
    vmm_sys_util::ioctl_iow_nr!(TUNSETVNETBE, TUNTAP, 222, c_int);
}

/// The parts of `struct ifreq` the tun driver looks at: the name and the flags, padded to the
/// size of the kernel's union.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Ifreq {
    name: [u8; IFNAMSIZ],
    flags: i16,
    _pad: [u8; 22],
}

impl Default for Ifreq {
    fn default() -> Self {
        Ifreq { name: [0; IFNAMSIZ], flags: 0, _pad: [0; 22] }
    }
}

impl Ifreq {
    fn set_name(&mut self, name: &str) {
        // pstrcpy(): at most IFNAMSIZ - 1 bytes and a terminating zero.
        let n = name.len().min(IFNAMSIZ - 1);
        self.name = [0; IFNAMSIZ];
        self.name[..n].copy_from_slice(&name.as_bytes()[..n]);
    }

    fn name(&self) -> String {
        let end = self.name.iter().position(|&b| b == 0).unwrap_or(IFNAMSIZ);
        String::from_utf8_lossy(&self.name[..end]).into_owned()
    }
}

/// An ioctl whose argument is a pointer to `arg`.
fn ioctl_ref<T>(fd: &impl AsRawFd, req: c_ulong, arg: &mut T) -> io::Result<()> {
    // SAFETY: every caller passes a request whose argument is a `T` the kernel reads or writes
    // in place: an int, an unsigned int or an `Ifreq`, which is as large as `struct ifreq`. The
    // reference is valid for the call and the fd is borrowed for its length.
    let r = unsafe { vmm_sys_util::ioctl::ioctl_with_mut_ref(fd, req, arg) };
    if r < 0 { Err(io::Error::last_os_error()) } else { Ok(()) }
}

/// An ioctl whose argument is a plain value.
fn ioctl_val(fd: &impl AsRawFd, req: c_ulong, arg: c_ulong) -> io::Result<()> {
    // SAFETY: TUNSETOFFLOAD, the one caller, takes its argument by value and touches no memory
    // of this process.
    let r = unsafe { vmm_sys_util::ioctl::ioctl_with_val(fd, req, arg) };
    if r < 0 { Err(io::Error::last_os_error()) } else { Ok(()) }
}

/// `if_nametoindex()`, through sysfs.
fn if_nametoindex(name: &str) -> Option<u32> {
    if name.is_empty() || name.contains('/') {
        return None;
    }
    let s = std::fs::read_to_string(format!("/sys/class/net/{name}/ifindex")).ok()?;
    s.trim().parse().ok().filter(|&i| i != 0)
}

fn open_rw(path: &str) -> io::Result<OwnedFd> {
    OpenOptions::new().read(true).write(true).open(path).map(OwnedFd::from)
}

/// `tap_open()`: makes or attaches to tap interface `ifname`, `tap%d` when it is empty, and
/// puts the name the kernel picked back into `ifname`. `vnet_hdr` says whether to ask for
/// virtio-net headers and comes back saying whether they are on.
pub(crate) fn tap_open(
    ifname: &mut String,
    vnet_hdr: &mut bool,
    vnet_hdr_required: bool,
    mq_required: bool,
) -> Result<OwnedFd> {
    let mut fd = None;
    if let Some(index) = if_nametoindex(ifname) {
        fd = open_rw(&format!("/dev/tap{index}")).ok();
    }
    let fd = match fd {
        Some(fd) => fd,
        None => loop {
            match open_rw(PATH_NET_TUN) {
                Ok(fd) => break fd,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    return Err(Error::generic(format!(
                        "Could not open '{PATH_NET_TUN}': {}",
                        strerror(&e)
                    )));
                }
            }
        },
    };

    let mut ifr = Ifreq { flags: IFF_TAP | IFF_NO_PI, ..Ifreq::default() };
    let mut features: u32 = 0;
    if let Err(e) = ioctl_ref(&fd, nr::TUNGETFEATURES(), &mut features) {
        warn_report(&format!("TUNGETFEATURES failed: {}", strerror(&e)));
        features = 0;
    }
    if features & IFF_ONE_QUEUE != 0 {
        ifr.flags |= IFF_ONE_QUEUE as i16;
    }
    if *vnet_hdr {
        *vnet_hdr = features & IFF_VNET_HDR != 0;
        if *vnet_hdr {
            ifr.flags |= IFF_VNET_HDR as i16;
        }
        if vnet_hdr_required && !*vnet_hdr {
            return Err(Error::generic(
                "vnet_hdr=1 requested, but no kernel support for IFF_VNET_HDR available",
            ));
        }
        // Reset the header size a persistent tap may have kept from someone else. Old kernels
        // do not know this ioctl, and then the size is right anyway.
        let mut len: i32 = crate::client::VNET_HDR_LEN as i32;
        let _ = ioctl_ref(&fd, nr::TUNSETVNETHDRSZ(), &mut len);
    }
    if mq_required {
        if features & IFF_MULTI_QUEUE == 0 {
            return Err(Error::generic(
                "multiqueue required, but no kernel support for IFF_MULTI_QUEUE available",
            ));
        }
        ifr.flags |= IFF_MULTI_QUEUE as i16;
    }
    let asked = if ifname.is_empty() { "tap%d" } else { ifname.as_str() };
    ifr.set_name(asked);
    if let Err(e) = ioctl_ref(&fd, nr::TUNSETIFF(), &mut ifr) {
        let msg = if ifname.is_empty() {
            format!("could not configure {PATH_NET_TUN}")
        } else {
            format!("could not configure {PATH_NET_TUN} ({})", ifr.name())
        };
        return Err(Error::from_io(msg, e));
    }
    *ifname = ifr.name();
    crate::poll::unblock(&fd)?;
    Ok(fd)
}

/// `tap_set_sndbuf()`.
pub(crate) fn set_sndbuf(fd: &impl AsRawFd, sndbuf: i32) -> Result<()> {
    let mut v = sndbuf;
    ioctl_ref(fd, nr::TUNSETSNDBUF(), &mut v)
        .map_err(|e| Error::from_io("TUNSETSNDBUF ioctl failed", e))
}

/// `tap_probe_vnet_hdr()`.
pub(crate) fn probe_vnet_hdr(fd: &impl AsRawFd) -> Result<bool> {
    let mut ifr = Ifreq::default();
    ioctl_ref(fd, nr::TUNGETIFF(), &mut ifr).map_err(|e| {
        Error::from_io(format!("Unable to query TUNGETIFF on FD {}", fd.as_raw_fd()), e)
    })?;
    Ok(ifr.flags as u16 as u32 & IFF_VNET_HDR != 0)
}

fn set_offload_bits(fd: &impl AsRawFd, bits: u32) -> io::Result<()> {
    ioctl_val(fd, nr::TUNSETOFFLOAD(), c_ulong::from(bits))
}

/// `tap_probe_has_ufo()`.
pub(crate) fn probe_has_ufo(fd: &impl AsRawFd) -> bool {
    set_offload_bits(fd, TUN_F_CSUM | TUN_F_UFO).is_ok()
}

/// `tap_probe_has_uso()`.
pub(crate) fn probe_has_uso(fd: &impl AsRawFd) -> bool {
    set_offload_bits(fd, TUN_F_CSUM | TUN_F_USO4 | TUN_F_USO6).is_ok()
}

/// `tap_probe_has_tunnel()`.
pub(crate) fn probe_has_tunnel(fd: &impl AsRawFd) -> bool {
    set_offload_bits(fd, TUN_F_CSUM | TUN_F_TSO4 | TUN_F_UDP_TUNNEL_GSO).is_ok()
}

/// `tap_fd_set_vnet_hdr_len()`.
pub(crate) fn set_vnet_hdr_len(fd: &impl AsRawFd, len: usize) {
    let mut v = len as i32;
    if let Err(e) = ioctl_ref(fd, nr::TUNSETVNETHDRSZ(), &mut v) {
        error_report(&format!("TUNSETVNETHDRSZ ioctl() failed: {}.", strerror(&e)));
    }
}

fn set_vnet_endian(fd: &impl AsRawFd, req: c_ulong, name: &str, on: bool) -> io::Result<()> {
    let mut arg: i32 = on.into();
    match ioctl_ref(fd, req, &mut arg) {
        Ok(()) => Ok(()),
        Err(e) if e.raw_os_error() == Some(libc::EINVAL) => Err(e),
        Err(e) => {
            // QEMU aborts here; the kernel refused something it claims to support.
            error_report(&format!("{name} ioctl() failed: {}.", strerror(&e)));
            Err(e)
        }
    }
}

/// `tap_fd_set_vnet_le()`.
pub(crate) fn set_vnet_le(fd: &impl AsRawFd, is_le: bool) -> io::Result<()> {
    set_vnet_endian(fd, nr::TUNSETVNETLE(), "TUNSETVNETLE", is_le)
}

/// `tap_fd_set_vnet_be()`.
pub(crate) fn set_vnet_be(fd: &impl AsRawFd, is_be: bool) -> io::Result<()> {
    set_vnet_endian(fd, nr::TUNSETVNETBE(), "TUNSETVNETBE", is_be)
}

/// `tap_fd_set_offload()`.
pub(crate) fn set_offload(fd: &impl AsRawFd, ol: &NetOffloads) {
    if let Err(e) = set_offload_bits(fd, 0) {
        if e.raw_os_error() == Some(libc::EINVAL) {
            return;
        }
    }
    let mut offload = 0;
    if ol.csum {
        offload |= TUN_F_CSUM;
        if ol.tso4 {
            offload |= TUN_F_TSO4;
        }
        if ol.tso6 {
            offload |= TUN_F_TSO6;
        }
        if (ol.tso4 || ol.tso6) && ol.ecn {
            offload |= TUN_F_TSO_ECN;
        }
        if ol.ufo {
            offload |= TUN_F_UFO;
        }
        if ol.uso4 {
            offload |= TUN_F_USO4;
        }
        if ol.uso6 {
            offload |= TUN_F_USO6;
        }
        if ol.tnl {
            offload |= TUN_F_UDP_TUNNEL_GSO;
        }
        if ol.tnl_csum {
            offload |= TUN_F_UDP_TUNNEL_GSO_CSUM;
        }
    }
    if set_offload_bits(fd, offload).is_err() {
        offload &= !(TUN_F_USO4 | TUN_F_USO6);
        if set_offload_bits(fd, offload).is_err() {
            offload &= !TUN_F_UFO;
            if let Err(e) = set_offload_bits(fd, offload) {
                eprintln!("TUNSETOFFLOAD ioctl() failed: {}", strerror(&e));
            }
        }
    }
}

fn set_queue(fd: &impl AsRawFd, flags: i16, what: &str) -> io::Result<()> {
    let mut ifr = Ifreq { flags, ..Ifreq::default() };
    ioctl_ref(fd, nr::TUNSETQUEUE(), &mut ifr).inspect_err(|_| {
        error_report(&format!("could not {what} queue"));
    })
}

/// `tap_fd_enable()`: attaches a queue of a multiqueue tap.
pub(crate) fn fd_enable(fd: &impl AsRawFd) -> io::Result<()> {
    set_queue(fd, IFF_ATTACH_QUEUE, "enable")
}

/// `tap_fd_disable()`: detaches a queue of a multiqueue tap.
pub(crate) fn fd_disable(fd: &impl AsRawFd) -> io::Result<()> {
    set_queue(fd, IFF_DETACH_QUEUE, "disable")
}

/// `tap_fd_get_ifname()`.
pub(crate) fn get_ifname(fd: &impl AsRawFd) -> io::Result<String> {
    let mut ifr = Ifreq::default();
    ioctl_ref(fd, nr::TUNGETIFF(), &mut ifr).inspect_err(|e| {
        error_report(&format!("TUNGETIFF ioctl() failed: {}", strerror(e)));
    })?;
    Ok(ifr.name())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ioctl_numbers_match_the_kernel() {
        assert_eq!(nr::TUNSETIFF(), 0x4004_54ca);
        assert_eq!(nr::TUNGETFEATURES(), 0x8004_54cf);
        assert_eq!(nr::TUNSETOFFLOAD(), 0x4004_54d0);
        assert_eq!(nr::TUNGETIFF(), 0x8004_54d2);
        assert_eq!(nr::TUNSETVNETHDRSZ(), 0x4004_54d8);
        assert_eq!(size_of::<Ifreq>(), 40);
    }
}
