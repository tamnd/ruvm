// SPDX-License-Identifier: MIT OR Apache-2.0

//! The kernel vhost frontend against the real `/dev/vhost-net` and `/dev/vhost-vsock`. Each test
//! skips itself when the device is missing or not accessible, which is the usual case in CI.

#![cfg(target_os = "linux")]

use std::os::fd::AsFd;
use std::os::unix::net::UnixStream;

use ruvm_vhost::kernel::VhostKernel;
use ruvm_vhost::{MemoryRegion, VhostBackend, VringAddr};

const VIRTIO_F_VERSION_1: u64 = 1 << 32;

fn open(path: &str) -> Option<VhostKernel> {
    match VhostKernel::open(path) {
        Ok(dev) => Some(dev),
        Err(e) => {
            eprintln!("skipping: cannot open {path}: {e}");
            None
        }
    }
}

#[test]
fn vhost_net_setup() {
    let Some(mut net) = open("/dev/vhost-net") else { return };
    // The generic steps go through the trait, the way the virtio device models will use it.
    let dev: &mut dyn VhostBackend = &mut net;
    dev.set_owner().unwrap();
    let features = dev.get_features().unwrap();
    assert_ne!(features & VIRTIO_F_VERSION_1, 0, "vhost-net offers VERSION_1");
    dev.set_features(features & VIRTIO_F_VERSION_1).unwrap();

    let mut memory = vec![0u64; 0x2000];
    let base = memory.as_mut_ptr() as u64;
    let region = MemoryRegion {
        guest_phys_addr: 0,
        memory_size: 0x10000,
        userspace_addr: base,
        mmap_offset: 0,
        fd: None,
    };
    dev.set_mem_table(&[region]).unwrap();
    dev.set_vring_num(0, 256).unwrap();
    dev.set_vring_base(0, 0).unwrap();
    let addr = VringAddr {
        index: 0,
        flags: 0,
        desc_user_addr: base,
        avail_user_addr: base + 0x1000,
        used_user_addr: base + 0x2000,
        log_guest_addr: 0,
    };
    dev.set_vring_addr(&addr).unwrap();
    dev.set_vring_call(0, None).unwrap();
    dev.set_vring_kick(0, None).unwrap();
    dev.set_vring_err(0, None).unwrap();
    let (a, _b) = UnixStream::pair().unwrap();
    // A socket is not a tap device, so attaching it fails, but detaching works.
    assert!(net.net_set_backend(0, Some(a.as_fd())).is_err());
    net.net_set_backend(0, None).unwrap();
    assert_eq!(net.get_vring_base(0).unwrap(), 0);
    net.reset_owner().unwrap();
}

#[test]
fn vhost_vsock_setup() {
    let Some(mut dev) = open("/dev/vhost-vsock") else { return };
    dev.set_owner().unwrap();
    // CID 2 is the host and can never be a guest's.
    assert!(dev.vsock_set_guest_cid(2).is_err());
    let cid = 0x1000_0000 + u64::from(std::process::id());
    if let Err(e) = dev.vsock_set_guest_cid(cid) {
        eprintln!("skipping: cannot set CID {cid}: {e}");
        return;
    }
    dev.vsock_set_running(false).unwrap();
}
